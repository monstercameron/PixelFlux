Execution providers go in this folder.

PixelFlux loads every .dll here through the runtime's plugin mechanism
(RegisterExecutionProviderLibrary) at startup and then asks for a class of device rather than a
named provider. That is why an accelerator is a file you copy rather than a build you rebuild, and
why one build runs on Qualcomm, AMD and Intel silicon.

A provider must be built against the same runtime ABI as the onnxruntime.dll shipped beside the
application, or it will be refused and logged. `pixelflux accel` prints what loaded and what the
runtime can see.

Known source, verified 2026-08-08: the Microsoft.Windows.AI.MachineLearning NuGet package ships a
win-arm64 onnxruntime.dll (1.27.1) with DirectML, QNN, WebGPU, VitisAI and OpenVINO names present,
plus DirectML.dll. On this machine only DirectML actually initialised.
