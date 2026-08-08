using Microsoft.Data.Sqlite;

namespace PixelFlux.Core.Index;

/// <summary>
/// Owns the local SQLite file: connections, pragmas, and schema migration.
/// </summary>
/// <remarks>
/// <para>
/// The database is the local index and search engine, not the source of truth. Originals on
/// disk are the truth; this file can be deleted and rebuilt by re-ingesting, at the cost of
/// losing whatever the user typed. That is why user-entered metadata is also published as
/// revision records to shared storage — see the sync layer.
/// </para>
/// <para>
/// One connection is opened per operation rather than pooling a long-lived one. SQLite in WAL
/// mode handles that well, and it keeps the threading story trivial: the ingestion pipeline,
/// the background worker, and the UI can all query concurrently without a shared lock to
/// reason about.
/// </para>
/// </remarks>
public sealed class PhotoDatabase
{
    /// <summary>Every migration, in order. Index in this array + 1 is the schema version.</summary>
    /// <remarks>
    /// Append-only. Never edit a shipped entry — a device that already ran it will not run it
    /// again, so an edit silently produces two different schemas in the same library.
    /// </remarks>
    private static readonly string[] Migrations =
    [
        // ---- v1: photos, tags, objects, embeddings, and the full-text index ----
        """
        CREATE TABLE photos (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash        TEXT    NOT NULL UNIQUE,
            perceptual_hash     TEXT    NOT NULL,
            original_path       TEXT    NOT NULL,
            file_name           TEXT    NOT NULL,
            mime_type           TEXT    NOT NULL,
            width               INTEGER NOT NULL DEFAULT 0,
            height              INTEGER NOT NULL DEFAULT 0,
            file_size           INTEGER NOT NULL DEFAULT 0,
            captured_utc        TEXT    NOT NULL,
            capture_exact       INTEGER NOT NULL DEFAULT 0,
            file_modified_utc   TEXT    NOT NULL,
            indexed_utc         TEXT    NOT NULL,
            camera_make         TEXT,
            camera_model        TEXT,
            lens_model          TEXT,
            iso                 INTEGER,
            f_number            REAL,
            exposure_seconds    REAL,
            focal_length_mm     REAL,
            gps_lat             REAL,
            gps_lon             REAL,
            gps_alt             REAL,
            orientation         INTEGER NOT NULL DEFAULT 1,
            thumbnail_key       TEXT,
            proxy_key           TEXT,
            state               INTEGER NOT NULL DEFAULT 0,
            state_detail        TEXT,
            model_version       TEXT,
            ai_caption          TEXT,
            ai_description      TEXT,
            user_title          TEXT,
            user_notes          TEXT,
            rating              INTEGER NOT NULL DEFAULT 0,
            is_favourite        INTEGER NOT NULL DEFAULT 0,
            revision            INTEGER NOT NULL DEFAULT 0
        );

        -- Chronology is the primary axis of the entire UI: the gallery sorts by it, the time
        -- rail buckets by it, and every date filter narrows on it. It gets the first index.
        CREATE INDEX ix_photos_captured   ON photos(captured_utc DESC);
        CREATE INDEX ix_photos_state      ON photos(state) WHERE state IN (0, 1);
        CREATE INDEX ix_photos_perceptual ON photos(perceptual_hash);
        CREATE INDEX ix_photos_camera     ON photos(camera_model) WHERE camera_model IS NOT NULL;

        CREATE TABLE photo_tags (
            photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
            tag        TEXT    NOT NULL,
            confidence REAL    NOT NULL DEFAULT 1.0,
            source     INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (photo_id, tag, source)
        );
        CREATE INDEX ix_tags_tag ON photo_tags(tag);

        CREATE TABLE photo_objects (
            photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
            label      TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL
        );
        CREATE INDEX ix_objects_label ON photo_objects(label);

        CREATE TABLE photo_embeddings (
            photo_id INTEGER NOT NULL PRIMARY KEY REFERENCES photos(id) ON DELETE CASCADE,
            model    TEXT    NOT NULL,
            dims     INTEGER NOT NULL,
            vector   BLOB    NOT NULL
        );
        """,

        // ---- v2: full-text index over everything textual, kept in step by triggers ----
        //
        // Deliberately an external-content-free FTS5 table rather than a contentless one.
        // A contentless table cannot be updated in place, and captions get rewritten every time
        // a model is re-run; carrying our own copy of the text costs a little disk and removes
        // a whole class of "the index disagrees with the row" bug.
        """
        CREATE VIRTUAL TABLE photo_fts USING fts5(
            title, caption, description, tags, filename, camera,
            tokenize = 'unicode61 remove_diacritics 2'
        );

        CREATE TABLE photo_fts_map (
            rowid    INTEGER PRIMARY KEY,
            photo_id INTEGER NOT NULL UNIQUE REFERENCES photos(id) ON DELETE CASCADE
        );
        """,

        // ---- v3: resolved view applying the user-beats-AI precedence rule in one place ----
        //
        // Every read path goes through this view so the precedence rule cannot drift between
        // the gallery, search, export, and the slideshow.
        """
        CREATE VIEW photos_resolved AS
        SELECT
            p.*,
            COALESCE(NULLIF(TRIM(p.user_title), ''), NULLIF(TRIM(p.ai_caption), ''), p.file_name)
                AS display_title,
            CASE
                WHEN TRIM(COALESCE(p.user_title, '')) <> '' THEN 2
                WHEN TRIM(COALESCE(p.ai_caption, '')) <> '' THEN 1
                ELSE 0
            END AS title_source
        FROM photos p;
        """,

        // ---- v4: collections — both hand-made albums and saved searches -------------------
        //
        // One table for two things that look different to the user and are nearly identical to
        // the system. A manual album is a fixed list of photo ids; a smart folder is a stored
        // query re-run on read. Keeping them in one table means the sidebar, rename, delete,
        // reorder, and export paths are written once, and a manual album can be turned into a
        // smart one (or the reverse) without moving rows between tables.
        """
        CREATE TABLE collections (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT    NOT NULL,
            kind         INTEGER NOT NULL DEFAULT 0,   -- 0 = manual, 1 = smart
            query_json   TEXT,                          -- populated for smart folders only
            cover_photo  INTEGER REFERENCES photos(id) ON DELETE SET NULL,
            position     INTEGER NOT NULL DEFAULT 0,
            created_utc  TEXT    NOT NULL,
            updated_utc  TEXT    NOT NULL
        );
        CREATE UNIQUE INDEX ux_collections_name ON collections(name COLLATE NOCASE);

        CREATE TABLE collection_photos (
            collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            photo_id      INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
            -- Manual albums are ordered by hand, and that order is the point of making one.
            position      INTEGER NOT NULL DEFAULT 0,
            added_utc     TEXT    NOT NULL,
            PRIMARY KEY (collection_id, photo_id)
        );
        CREATE INDEX ix_collection_photos_photo ON collection_photos(photo_id);

        -- The folder each photo was imported from, denormalised out of original_path.
        --
        -- Derivable with string surgery on every query, but people organise photographs in
        -- folders and then expect to browse by them, so this is a first-class axis rather than
        -- something reconstructed. Indexed because the sidebar groups by it on every load.
        ALTER TABLE photos ADD COLUMN source_folder TEXT NOT NULL DEFAULT '';
        CREATE INDEX ix_photos_folder ON photos(source_folder);
        """,

        // ---- v5: resolved place names -------------------------------------------------
        //
        // Coordinates are useless to a person. These columns hold what a GPS fix actually
        // means — a city and a country — resolved once at ingest from the embedded gazetteer
        // rather than on every render.
        //
        // Denormalised on purpose. Place is a browsing axis: the facet groups by it, search
        // matches on it, and the viewer prints it. Recomputing a nearest-neighbour lookup for
        // every row of every query to avoid three columns would be a poor trade.
        """
        ALTER TABLE photos ADD COLUMN place_city    TEXT;
        ALTER TABLE photos ADD COLUMN place_country TEXT;
        ALTER TABLE photos ADD COLUMN place_code    TEXT;
        ALTER TABLE photos ADD COLUMN place_label   TEXT;

        CREATE INDEX ix_photos_country ON photos(place_country) WHERE place_country IS NOT NULL;
        CREATE INDEX ix_photos_city    ON photos(place_city)    WHERE place_city    IS NOT NULL;
        """,

        // ---- v6: segmentation ------------------------------------------------------------
        //
        // What the model found in each photograph, and where. Separate from photo_objects
        // (which holds plain boxes) because a segment carries a mask and an honest area, and
        // because these rows are wholly owned by a model version: re-running a better model
        // deletes and rewrites them, which must not touch anything a person wrote.
        //
        // The mask itself is NOT in the database. A 256-pixel mask is a few kilobytes, and at
        // twenty segments across fifty thousand photographs that is a million blobs turning the
        // index from a fast queryable file into a slow binary store. Masks are written to the
        // derivative cache as greyscale PNGs and referenced by key, exactly like thumbnails —
        // the WebView then loads them as ordinary images and CSS tints them.
        """
        CREATE TABLE photo_segments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
            label       TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
            area        REAL    NOT NULL,
            prominence  REAL    NOT NULL,
            mask_key    TEXT,
            model       TEXT    NOT NULL
        );

        CREATE INDEX ix_segments_photo ON photo_segments(photo_id);
        CREATE INDEX ix_segments_label ON photo_segments(label);
        -- The object facet reads label + prominence together on every sidebar render.
        CREATE INDEX ix_segments_rank  ON photo_segments(label, prominence DESC);
        """,

        // ---- v7: faces -------------------------------------------------------------------
        //
        // Its own table rather than rows in photo_segments, for three reasons that all point
        // the same way. A face has geometry a segment does not — five landmarks and a roll
        // angle, which is what lets a crop be straightened now and a recognition model align
        // one later. A face carries no label, so a "person" row in a labelled table would be
        // a permanent lie. And the browsing question is different: photo_segments answers
        // "what is in this photograph", faces answers "who appears in this library", which
        // reads the table the other way round and wants its own index for it.
        //
        // Landmarks are ten comma-separated fractions in one column, not ten columns or a
        // child table. Nothing queries an individual eye; they are read back whole or not at
        // all, and the alternatives cost either ten columns of noise or a join per face.
        //
        // Crops live in the derivative cache like every other derived image, keyed by content
        // hash so they survive the index being rebuilt.
        //
        // No person_id here. Grouping faces by identity needs a recognition model and a
        // clustering pass, neither of which exists yet, and a column reserved for a design
        // that has not been made is a column that will turn out to be the wrong shape.
        """
        CREATE TABLE photo_faces (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
            confidence  REAL    NOT NULL,
            x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
            area        REAL    NOT NULL,
            roll        REAL    NOT NULL,
            landmarks   TEXT    NOT NULL,
            crop_key    TEXT,
            model       TEXT    NOT NULL
        );

        CREATE INDEX ix_faces_photo ON photo_faces(photo_id);
        -- The faces page is one query: every face, biggest first. This is that query's index.
        CREATE INDEX ix_faces_area  ON photo_faces(area DESC);
        """,

        // ---- v8: face embeddings ---------------------------------------------------------
        //
        // The vector that makes "show me this person" possible: 128 floats describing a face
        // such that two photographs of one person land close together.
        //
        // A BLOB on the face row rather than a table of floats. Nothing ever queries a single
        // dimension — the only operation is "compare this whole vector against every other
        // whole vector" — so a row per dimension would be 128 times the rows for no query it
        // enables. 512 bytes per face is 5 MB across ten thousand faces, which sits in memory
        // comfortably and is why brute-force comparison is the right algorithm here rather than
        // an approximate index.
        //
        // The model is recorded alongside, because vectors from two models are not comparable
        // and silently mixing them would produce confident nonsense: every face would look
        // slightly unlike every other, and the feature would appear to work while matching
        // nothing.
        """
        ALTER TABLE photo_faces ADD COLUMN embedding   BLOB;
        ALTER TABLE photo_faces ADD COLUMN embed_model TEXT;

        -- The comparison sweep reads every embedded face and nothing else.
        CREATE INDEX ix_faces_embedded ON photo_faces(embed_model) WHERE embedding IS NOT NULL;
        """,

        // ---- v9: corrections a person made to what the model said ------------------------
        //
        // The model calls a red sports car a truck and a cat's cardboard box a bed. Being able
        // to say "no, that is a car" is the difference between a tool that guesses at your
        // library and one you can actually rely on to find things.
        //
        // A separate column rather than overwriting `label`, because the two are different
        // kinds of claim and the difference has to survive. `label` is what the model said and
        // belongs to the model version; `user_label` is what a person said and belongs to the
        // person. Overwriting would lose the ability to re-run a better model and see whether
        // it now agrees, and it would make a correction indistinguishable from a detection.
        //
        // The hue, the search index, and the object facet all follow the user's word where
        // there is one — a correction that only changed a caption would not be worth making.
        """
        ALTER TABLE photo_segments ADD COLUMN user_label TEXT;

        CREATE INDEX ix_segments_userlabel ON photo_segments(user_label)
            WHERE user_label IS NOT NULL;
        """,

        // ---- v10: the analysis queue, its cache, and somewhere to keep settings -----------
        //
        // Four analyses now run over every photograph — a vision model writes a description, a
        // segmenter finds objects, a detector finds faces, an encoder makes the search vector —
        // and until now each swept the library independently, each deciding on its own what was
        // outstanding. That arrangement has three faults, and this table fixes all three.
        //
        // It could not express ORDER. The search vector is better when it can read the
        // description, so describing has to happen first; nothing enforced that, and whichever
        // sweep a person happened to start won.
        //
        // It could not RESUME. Progress lived in whatever the sweep had already written, so a
        // stage that failed halfway through a photograph left no record of having tried, and a
        // stage that failed for a reason that will never go away — a truncated file — was
        // retried on every single sweep, forever.
        //
        // It could not be PACED. A sweep ran flat out until it finished. On a laptop that is the
        // difference between a machine you can use and a machine you cannot, and the work is not
        // urgent: nobody is waiting on the description of a photograph taken in 2014.
        //
        // One row per photograph per stage, so the queue is a table you can look at rather than
        // a state machine you have to infer. `state` is where it got to, `model` is what produced
        // the current answer — a newer model makes the row outstanding again without any
        // migration — and `attempts` is what stops a permanently broken file from being retried
        // for the rest of the library's life.
        """
        CREATE TABLE photo_jobs (
            photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
            stage      TEXT    NOT NULL,
            -- The stage's position in the sequence, denormalised onto the row. The runner's claim
            -- query has to ask "is every earlier stage finished for this photograph", and a
            -- number compares in SQL where an enum name does not. Writing a CASE over four
            -- spellings in every query that needs the order is the alternative, and it is the
            -- kind of thing that gets updated in three places out of four.
            ord        INTEGER NOT NULL,
            state      TEXT    NOT NULL,
            model      TEXT,
            attempts   INTEGER NOT NULL DEFAULT 0,
            error      TEXT,
            updated_at TEXT    NOT NULL,
            PRIMARY KEY (photo_id, stage)
        ) WITHOUT ROWID;

        -- The runner's only question, asked once per item: what is outstanding? Partial, because
        -- a finished library is almost entirely 'done' rows and there is no reason to carry them
        -- in the index the runner reads.
        CREATE INDEX ix_jobs_outstanding ON photo_jobs(ord, photo_id)
            WHERE state IN ('pending', 'failed');

        -- Two supporting reads: the per-photograph ordering check the claim query runs, and the
        -- status readout asking how far along each stage is.
        CREATE INDEX ix_jobs_photo_ord   ON photo_jobs(photo_id, ord);
        CREATE INDEX ix_jobs_stage_state ON photo_jobs(stage, state);

        -- Results keyed by what was analysed rather than by which row happens to hold it.
        --
        -- A photograph costs roughly sixteen seconds of vision model to describe. Import the same
        -- file twice, keep a copy in a second folder, or rebuild the index after changing
        -- something unrelated, and every one of those seconds is spent again for an answer that
        -- cannot have changed: the input was the same bytes and the model was the same model.
        --
        -- Keying on the content hash rather than the photo id is the whole point. The hash is a
        -- property of the image; the id is a property of this database. Two rows that are the
        -- same picture share a cache entry, and an index rebuilt from scratch reuses everything.
        --
        -- The model is part of the key, not a column to compare against. Entries from an older
        -- model stay readable — useful for showing what changed — and are simply never hit once
        -- the runner starts asking for a newer one.
        CREATE TABLE stage_cache (
            content_hash TEXT NOT NULL,
            stage        TEXT NOT NULL,
            model        TEXT NOT NULL,
            payload      TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            PRIMARY KEY (content_hash, stage, model)
        ) WITHOUT ROWID;

        -- Settings that belong to the library, not to the machine looking at it.
        --
        -- When the analysis queue is allowed to run is a property of the photo library — put the
        -- library on a shared drive and the schedule should travel with it — so it goes here
        -- rather than in per-machine application storage. Key and value, because the alternative
        -- is a migration every time a preference is added, and preferences are exactly the thing
        -- that gets added casually.
        CREATE TABLE app_settings (
            key   TEXT NOT NULL PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        """,

        // ---- v11: people, and the faces that belong to them -------------------------------
        //
        // Until now the faces table deliberately had no identity column, and the comment on v7
        // said why: grouping by appearance is a guess, and reserving a column for a design that
        // had not been made would produce the wrong shape. The design has now been made, and it
        // is not the one that comment was worried about — this is not a model's opinion about who
        // somebody is, it is a name a person typed.
        //
        // That distinction is the whole table. Recognition still groups by similarity and is
        // still recomputed on demand; a person_id is a fact, set deliberately, and it outranks
        // anything the model thinks. The two are allowed to disagree.
        //
        // Names are unique and case-insensitive, so "Mum" and "mum" are one person rather than
        // two halves of a collection. That is a real constraint rather than a tidiness one:
        // duplicates here would split somebody's photographs in half with no visible cause.
        """
        CREATE TABLE people (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
            created_utc  TEXT    NOT NULL
        );

        -- ON DELETE SET NULL, not CASCADE. Deleting a person means "I no longer want this name",
        -- not "delete these faces" — the faces are still in the photographs either way, and
        -- cascading would quietly remove detections a model found.
        ALTER TABLE photo_faces ADD COLUMN person_id INTEGER REFERENCES people(id) ON DELETE SET NULL;

        -- The query behind "show me every photograph of this person".
        CREATE INDEX ix_faces_person ON photo_faces(person_id) WHERE person_id IS NOT NULL;
        """,
    ];

    private readonly string _connectionString;

    /// <summary>Creates a database handle for a SQLite file, creating the file if needed.</summary>
    /// <param name="databasePath">Absolute path to the <c>.db</c> file.</param>
    public PhotoDatabase(string databasePath)
    {
        if (string.IsNullOrWhiteSpace(databasePath))
        {
            throw new ArgumentException("A database path is required.", nameof(databasePath));
        }

        DatabasePath = Path.GetFullPath(databasePath);
        Directory.CreateDirectory(Path.GetDirectoryName(DatabasePath)!);

        _connectionString = new SqliteConnectionStringBuilder
        {
            DataSource = DatabasePath,
            Mode = SqliteOpenMode.ReadWriteCreate,
            Cache = SqliteCacheMode.Shared,
            Pooling = true,
        }.ToString();
    }

    /// <summary>Absolute path to the SQLite file.</summary>
    public string DatabasePath { get; }

    /// <summary>The schema version this build expects.</summary>
    public static int TargetSchemaVersion => Migrations.Length;

    /// <summary>Opens a configured connection. The caller owns and must dispose it.</summary>
    /// <returns>An open connection with PixelFlux's pragmas applied.</returns>
    public SqliteConnection Open()
    {
        var connection = new SqliteConnection(_connectionString);
        connection.Open();

        using SqliteCommand pragma = connection.CreateCommand();
        // WAL: readers never block the writer, which is what lets the gallery stay responsive
        //      while a background worker is committing analysis results.
        // NORMAL: the durability step down from FULL. A power cut can lose the last few
        //      commits, which for a rebuildable index is an acceptable trade for the write
        //      throughput ingestion needs.
        // foreign_keys: off by default in SQLite; the ON DELETE CASCADE rules above are inert
        //      without it, and orphaned tags would accumulate silently.
        pragma.CommandText = """
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA foreign_keys = ON;
            PRAGMA busy_timeout = 5000;
            PRAGMA temp_store = MEMORY;
            """;
        pragma.ExecuteNonQuery();

        return connection;
    }

    /// <summary>
    /// Brings the schema up to <see cref="TargetSchemaVersion"/>, applying any missing
    /// migrations in order. Safe to call on every startup.
    /// </summary>
    /// <returns>The schema version before migration ran, so callers can log an upgrade.</returns>
    public int Migrate()
    {
        using SqliteConnection connection = Open();

        using (SqliteCommand read = connection.CreateCommand())
        {
            read.CommandText = "PRAGMA user_version;";
            int current = Convert.ToInt32(read.ExecuteScalar(), System.Globalization.CultureInfo.InvariantCulture);

            if (current >= Migrations.Length)
            {
                return current;
            }

            // Each migration runs in its own transaction. A failure therefore leaves the
            // database at the last complete version rather than half-way through a step,
            // and the next startup retries from exactly there.
            for (int version = current; version < Migrations.Length; version++)
            {
                using SqliteTransaction tx = connection.BeginTransaction();
                using (SqliteCommand apply = connection.CreateCommand())
                {
                    apply.Transaction = tx;
                    apply.CommandText = Migrations[version];
                    apply.ExecuteNonQuery();
                }

                using (SqliteCommand stamp = connection.CreateCommand())
                {
                    stamp.Transaction = tx;
                    // PRAGMA user_version does not accept a parameter, and the value is an
                    // int from a private array rather than anything caller-supplied.
                    stamp.CommandText = $"PRAGMA user_version = {version + 1};";
                    stamp.ExecuteNonQuery();
                }

                tx.Commit();
            }

            return current;
        }
    }
}
