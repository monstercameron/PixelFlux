// The only JavaScript in PixelFlux.
//
// Blazor handles the entire UI; what lives here is the work that genuinely cannot cross the
// bridge: reading layout the browser has already computed, and high-frequency pointer and wheel
// handling where a round trip per event would be unusable.

window.pixelflux = {

    /**
     * How many columns the photo grid is currently showing.
     *
     * The grid is `repeat(auto-fill, minmax(var(--cell), 1fr))`, so the column count is decided
     * by the browser from the available width and changes on every resize and every zoom step.
     * Keyboard navigation needs that number — pressing Down must move one row, and a row is
     * however many cells the browser decided to fit. Recomputing it in C# would mean duplicating
     * the CSS layout algorithm and getting it subtly wrong.
     */
    gridColumns: function (element) {
        if (!element) { return 1; }
        const columns = getComputedStyle(element).gridTemplateColumns;
        if (!columns || columns === 'none') { return 1; }
        return Math.max(1, columns.split(' ').filter(c => c.trim().length > 0).length);
    },

    /**
     * Scroll a grid cell into view without yanking the page around.
     *
     * `block: 'nearest'` is the important part: it scrolls only when the cell is actually off
     * screen. Using 'center' instead makes every arrow keypress re-centre the view, so holding
     * an arrow key turns the grid into a slot machine.
     */
    revealCell: function (container, index) {
        if (!container) { return; }
        const cell = container.children[index];
        if (cell && cell.scrollIntoView) {
            cell.scrollIntoView({ block: 'nearest', inline: 'nearest' });
        }
    },

    /**
     * Move focus to an element by id, if it is present.
     *
     * Used for the `/` shortcut and for returning focus to the grid when the viewer closes.
     * Focus that vanishes after a dialog closes is the single most common way a keyboard user
     * gets stranded in an application.
     */
    focusId: function (id) {
        const element = document.getElementById(id);
        if (element) { element.focus(); }
    },

    /**
     * Stop the browser from zooming the whole interface.
     *
     * A photo viewer that has its own zoom must not also have the page zoom underneath it, or a
     * pinch does both at once and the chrome ends up at 140% with no obvious way back. Three
     * routes have to be closed:
     *
     *   - trackpad pinch, which arrives as a wheel event with ctrlKey set
     *   - Ctrl+plus / Ctrl+minus / Ctrl+0
     *   - Ctrl+wheel on a mouse
     *
     * The WebView's own pinch-zoom setting is disabled separately in MainPage; this covers the
     * routes that setting does not.
     */
    blockPageZoom: function () {
        window.addEventListener('wheel', e => {
            if (e.ctrlKey) { e.preventDefault(); }
        }, { passive: false, capture: true });

        window.addEventListener('keydown', e => {
            if ((e.ctrlKey || e.metaKey) && ['+', '=', '-', '_', '0'].includes(e.key)) {
                e.preventDefault();
            }
        }, { capture: true });

        // Safari/WebKit-style gesture events, harmless to guard for.
        ['gesturestart', 'gesturechange', 'gestureend'].forEach(name =>
            window.addEventListener(name, e => e.preventDefault(), { passive: false }));
    },

    /**
     * Pinch-to-zoom and two-finger pan, scoped to one photograph.
     *
     * INPUT ROUTES
     * ------------
     * The same gesture arrives three different ways depending on hardware, and all three have to
     * be handled or the feature works on one machine and not another:
     *
     *   - precision trackpad pinch  -> wheel events with ctrlKey set (NOT touch events)
     *   - precision trackpad scroll -> wheel events with deltaX/deltaY, which is the two-finger pan
     *   - touchscreen               -> two simultaneous pointer events, distance between them
     *   - mouse                     -> wheel to zoom, drag to pan
     *
     * The transform is applied to the wrapper, not the image, so the segmentation overlay zooms
     * and pans locked to the photograph. Transforming them separately would drift the masks off
     * their subjects the moment anyone zoomed in — which is exactly when a mask matters most.
     */
    zoom: (function () {
        let el = null;              // the element being transformed
        let scale = 1, tx = 0, ty = 0;
        let pointers = new Map();   // active touch pointers, for pinch
        let pinchStart = 0, pinchScale = 1;
        let dragging = false, lastX = 0, lastY = 0;
        let onChange = null;
        let handlers = [];

        const MIN = 1, MAX = 8;

        function apply() {
            if (!el) { return; }
            el.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
            el.style.cursor = scale > 1 ? (dragging ? 'grabbing' : 'grab') : '';
            if (onChange) { onChange(Math.round(scale * 100)); }
        }

        /**
         * Keep the photograph from being panned off the screen.
         *
         * At scale 1 it is pinned centred — there is nothing to pan to, and allowing it would let
         * someone flick the picture away and be unable to find it. Beyond 1, travel is limited to
         * the overhang, so an edge can reach the edge of the frame and no further.
         */
        function clamp() {
            if (!el) { return; }
            if (scale <= 1) { scale = 1; tx = 0; ty = 0; return; }

            // getBoundingClientRect reports the element AFTER the transform, so dividing by the
            // scale recovers its laid-out size. The overhang — how far the enlarged picture
            // sticks out past its own frame on one side — is half the growth, not the whole of
            // it. The first version added the same quantity twice and let the photograph be
            // dragged until half of it was outside the frame with black filling the rest.
            const r = el.getBoundingClientRect();
            const base = { w: r.width / scale, h: r.height / scale };

            const overX = (base.w * (scale - 1)) / 2;
            const overY = (base.h * (scale - 1)) / 2;

            tx = Math.min(overX, Math.max(-overX, tx));
            ty = Math.min(overY, Math.max(-overY, ty));
        }

        /** Zoom by a factor, keeping the point under the cursor fixed. */
        function zoomAt(factor, clientX, clientY) {
            if (!el) { return; }
            const before = scale;
            scale = Math.min(MAX, Math.max(MIN, scale * factor));
            if (scale === before) { return; }

            // Anchor the cursor: the content point under it must not move. Solving for the new
            // translation is what makes zoom feel like it is following the pointer rather than
            // the centre of the frame.
            const r = el.getBoundingClientRect();
            const px = clientX - r.left - r.width / 2;
            const py = clientY - r.top - r.height / 2;
            const ratio = scale / before;

            tx = tx - px * (ratio - 1);
            ty = ty - py * (ratio - 1);

            clamp();
            apply();
        }

        function on(target, name, fn, opts) {
            target.addEventListener(name, fn, opts);
            handlers.push([target, name, fn, opts]);
        }

        return {
            attach: function (element, dotnet) {
                this.detach();
                if (!element) { return; }

                el = element;
                scale = 1; tx = 0; ty = 0;
                el.style.transformOrigin = 'center center';
                el.style.touchAction = 'none';   // we handle every gesture ourselves

                onChange = dotnet
                    ? p => dotnet.invokeMethodAsync('OnZoomChanged', p)
                    : null;

                on(el, 'wheel', e => {
                    e.preventDefault();

                    if (e.ctrlKey) {
                        // Trackpad pinch, or Ctrl+wheel on a mouse.
                        //
                        // Two things made the first version coarse. deltaMode was ignored, so a
                        // device reporting in lines or pages was multiplied as though it had
                        // reported pixels; and the 0.01 coefficient turned one notch of a mouse
                        // wheel — 100 units — into a 2.7x jump. Normalising the units first and
                        // then applying a much gentler exponent gives a pinch dozens of small
                        // steps across its travel instead of three big ones, and still leaves a
                        // mouse notch at a useful ~13%.
                        const units = e.deltaMode === 1 ? e.deltaY * 16
                                    : e.deltaMode === 2 ? e.deltaY * 100
                                    : e.deltaY;

                        // Clamped per event so one violent flick cannot cross the whole range.
                        const step = Math.max(-40, Math.min(40, units));
                        zoomAt(Math.exp(-step * 0.0032), e.clientX, e.clientY);
                    } else if (scale > 1) {
                        // Two-finger pan. Only meaningful when zoomed in — at 1:1 there is
                        // nowhere to go, and swallowing the gesture there would feel broken.
                        tx -= e.deltaX;
                        ty -= e.deltaY;
                        clamp();
                        apply();
                    }
                }, { passive: false });

                on(el, 'pointerdown', e => {
                    el.setPointerCapture(e.pointerId);
                    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

                    if (pointers.size === 2) {
                        const [a, b] = [...pointers.values()];
                        pinchStart = Math.hypot(a.x - b.x, a.y - b.y);
                        pinchScale = scale;
                    } else if (scale > 1) {
                        dragging = true;
                        lastX = e.clientX;
                        lastY = e.clientY;
                        apply();
                    }
                });

                on(el, 'pointermove', e => {
                    if (!pointers.has(e.pointerId)) { return; }
                    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

                    if (pointers.size === 2) {
                        const [a, b] = [...pointers.values()];
                        const dist = Math.hypot(a.x - b.x, a.y - b.y);
                        if (pinchStart > 0) {
                            const target = pinchScale * (dist / pinchStart);
                            zoomAt(target / scale, (a.x + b.x) / 2, (a.y + b.y) / 2);
                        }
                    } else if (dragging) {
                        tx += e.clientX - lastX;
                        ty += e.clientY - lastY;
                        lastX = e.clientX;
                        lastY = e.clientY;
                        clamp();
                        apply();
                    }
                });

                const release = e => {
                    pointers.delete(e.pointerId);
                    if (pointers.size < 2) { pinchStart = 0; }
                    if (pointers.size === 0) { dragging = false; apply(); }
                };

                on(el, 'pointerup', release);
                on(el, 'pointercancel', release);

                // Double-click toggles between fit and 2x, at the point clicked. The fastest way
                // in and back out, and what every image viewer does.
                on(el, 'dblclick', e => {
                    e.preventDefault();
                    if (scale > 1) { this.reset(); }
                    else { zoomAt(2, e.clientX, e.clientY); }
                });

                apply();
            },

            reset: function () {
                scale = 1; tx = 0; ty = 0;
                apply();
            },

            /** Zoom a step from a keyboard or a button, anchored at the centre. */
            step: function (factor) {
                if (!el) { return; }
                const r = el.getBoundingClientRect();
                zoomAt(factor, r.left + r.width / 2, r.top + r.height / 2);
            },

            detach: function () {
                handlers.forEach(([t, n, f, o]) => t.removeEventListener(n, f, o));
                handlers = [];
                pointers.clear();
                dragging = false;
                el = null;
                onChange = null;
            }
        };
    })()
};

window.pixelflux.blockPageZoom();
