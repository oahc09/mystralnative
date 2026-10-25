/**
 * Canvas 2D JavaScript Bindings
 *
 * Creates JavaScript objects that wrap native Canvas2DContext.
 * This exposes the CanvasRenderingContext2D API to JavaScript.
 */

#include "mystral/canvas/canvas2d.h"
#include "mystral/js/engine.h"
#include <iostream>
#include <cmath>
#include <unordered_map>

namespace mystral {
namespace canvas {

// Global storage for Canvas2D contexts (prevents them from being destroyed)
static std::unordered_map<void*, std::unique_ptr<Canvas2DContext>> g_canvas2dContexts;

// Store reference to JS engine for callbacks
static js::Engine* g_jsEngine = nullptr;

/**
 * Create a CanvasRenderingContext2D JS object that wraps a native Canvas2DContext
 *
 * IMPORTANT: Each method captures the native context pointer in its closure,
 * allowing multiple canvas contexts to work independently. This fixes the bug
 * where a global __canvas2dContext variable was used, causing only the last
 * created canvas to work.
 */
js::JSValueHandle createCanvas2DJSObject(js::Engine* engine, Canvas2DContext* ctx) {
    g_jsEngine = engine;

    auto jsCtx = engine->newObject();

    // Store the native context pointer
    engine->setPrivateData(jsCtx, ctx);

    // Mark the type
    engine->setProperty(jsCtx, "_contextType", engine->newString("2d"));

    // Capture the native context pointer for use in all method closures
    // This ensures each canvas context object has methods that use its own context
    Canvas2DContext* capturedCtx = ctx;

    // ========================================================================
    // Properties (as getter functions for now)
    // ========================================================================

    // canvas property (will be set by caller)
    engine->setProperty(jsCtx, "canvas", engine->newNull());

    // fillStyle
    engine->setProperty(jsCtx, "fillStyle", engine->newString("#000000"));
    engine->setProperty(jsCtx, "_setFillStyle",
        engine->newFunction("_setFillStyle", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && !args.empty()) {
                capturedCtx->setFillStyle(g_jsEngine->toString(args[0]));
            }
            return g_jsEngine->newUndefined();
        })
    );

    // strokeStyle
    engine->setProperty(jsCtx, "strokeStyle", engine->newString("#000000"));
    engine->setProperty(jsCtx, "_setStrokeStyle",
        engine->newFunction("_setStrokeStyle", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && !args.empty()) {
                capturedCtx->setStrokeStyle(g_jsEngine->toString(args[0]));
            }
            return g_jsEngine->newUndefined();
        })
    );

    // lineWidth
    engine->setProperty(jsCtx, "lineWidth", engine->newNumber(1.0));
    engine->setProperty(jsCtx, "_setLineWidth",
        engine->newFunction("_setLineWidth", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && !args.empty()) {
                capturedCtx->setLineWidth(static_cast<float>(g_jsEngine->toNumber(args[0])));
            }
            return g_jsEngine->newUndefined();
        })
    );

    // globalAlpha
    engine->setProperty(jsCtx, "globalAlpha", engine->newNumber(1.0));
    engine->setProperty(jsCtx, "_setGlobalAlpha",
        engine->newFunction("_setGlobalAlpha", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && !args.empty()) {
                capturedCtx->setGlobalAlpha(static_cast<float>(g_jsEngine->toNumber(args[0])));
            }
            return g_jsEngine->newUndefined();
        })
    );

    // font
    engine->setProperty(jsCtx, "font", engine->newString("10px sans-serif"));
    engine->setProperty(jsCtx, "_setFont",
        engine->newFunction("_setFont", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && !args.empty()) {
                capturedCtx->setFont(g_jsEngine->toString(args[0]));
            }
            return g_jsEngine->newUndefined();
        })
    );

    // textAlign
    engine->setProperty(jsCtx, "textAlign", engine->newString("start"));
    engine->setProperty(jsCtx, "_setTextAlign",
        engine->newFunction("_setTextAlign", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && !args.empty()) {
                capturedCtx->setTextAlign(g_jsEngine->toString(args[0]));
            }
            return g_jsEngine->newUndefined();
        })
    );

    // textBaseline
    engine->setProperty(jsCtx, "textBaseline", engine->newString("alphabetic"));
    engine->setProperty(jsCtx, "_setTextBaseline",
        engine->newFunction("_setTextBaseline", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && !args.empty()) {
                capturedCtx->setTextBaseline(g_jsEngine->toString(args[0]));
            }
            return g_jsEngine->newUndefined();
        })
    );

    // ========================================================================
    // Methods - all capture the native context pointer in their closure
    // ========================================================================

    // save()
    engine->setProperty(jsCtx, "save",
        engine->newFunction("save", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx) capturedCtx->save();
            return g_jsEngine->newUndefined();
        })
    );

    // restore()
    engine->setProperty(jsCtx, "restore",
        engine->newFunction("restore", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx) capturedCtx->restore();
            return g_jsEngine->newUndefined();
        })
    );

    // fillText(text, x, y)
    engine->setProperty(jsCtx, "fillText",
        engine->newFunction("fillText", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 3) {
                std::string text = g_jsEngine->toString(args[0]);
                float x = static_cast<float>(g_jsEngine->toNumber(args[1]));
                float y = static_cast<float>(g_jsEngine->toNumber(args[2]));
                capturedCtx->fillText(text, x, y);
            }
            return g_jsEngine->newUndefined();
        })
    );

    // strokeText(text, x, y)
    engine->setProperty(jsCtx, "strokeText",
        engine->newFunction("strokeText", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 3) {
                std::string text = g_jsEngine->toString(args[0]);
                float x = static_cast<float>(g_jsEngine->toNumber(args[1]));
                float y = static_cast<float>(g_jsEngine->toNumber(args[2]));
                capturedCtx->strokeText(text, x, y);
            }
            return g_jsEngine->newUndefined();
        })
    );

    // measureText(text) -> { width }
    engine->setProperty(jsCtx, "measureText",
        engine->newFunction("measureText", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            auto result = g_jsEngine->newObject();
            if (capturedCtx && !args.empty()) {
                std::string text = g_jsEngine->toString(args[0]);
                TextMetrics metrics = capturedCtx->measureText(text);

                g_jsEngine->setProperty(result, "width", g_jsEngine->newNumber(metrics.width));
                g_jsEngine->setProperty(result, "actualBoundingBoxLeft", g_jsEngine->newNumber(metrics.actualBoundingBoxLeft));
                g_jsEngine->setProperty(result, "actualBoundingBoxRight", g_jsEngine->newNumber(metrics.actualBoundingBoxRight));
                g_jsEngine->setProperty(result, "actualBoundingBoxAscent", g_jsEngine->newNumber(metrics.actualBoundingBoxAscent));
                g_jsEngine->setProperty(result, "actualBoundingBoxDescent", g_jsEngine->newNumber(metrics.actualBoundingBoxDescent));
                g_jsEngine->setProperty(result, "fontBoundingBoxAscent", g_jsEngine->newNumber(metrics.fontBoundingBoxAscent));
                g_jsEngine->setProperty(result, "fontBoundingBoxDescent", g_jsEngine->newNumber(metrics.fontBoundingBoxDescent));
            } else {
                g_jsEngine->setProperty(result, "width", g_jsEngine->newNumber(0));
            }
            return result;
        })
    );

    // fillRect(x, y, width, height)
    engine->setProperty(jsCtx, "fillRect",
        engine->newFunction("fillRect", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 4) {
                capturedCtx->fillRect(
                    static_cast<float>(g_jsEngine->toNumber(args[0])),
                    static_cast<float>(g_jsEngine->toNumber(args[1])),
                    static_cast<float>(g_jsEngine->toNumber(args[2])),
                    static_cast<float>(g_jsEngine->toNumber(args[3]))
                );
            }
            return g_jsEngine->newUndefined();
        })
    );

    // strokeRect(x, y, width, height)
    engine->setProperty(jsCtx, "strokeRect",
        engine->newFunction("strokeRect", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 4) {
                capturedCtx->strokeRect(
                    static_cast<float>(g_jsEngine->toNumber(args[0])),
                    static_cast<float>(g_jsEngine->toNumber(args[1])),
                    static_cast<float>(g_jsEngine->toNumber(args[2])),
                    static_cast<float>(g_jsEngine->toNumber(args[3]))
                );
            }
            return g_jsEngine->newUndefined();
        })
    );

    // clearRect(x, y, width, height)
    engine->setProperty(jsCtx, "clearRect",
        engine->newFunction("clearRect", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 4) {
                capturedCtx->clearRect(
                    static_cast<float>(g_jsEngine->toNumber(args[0])),
                    static_cast<float>(g_jsEngine->toNumber(args[1])),
                    static_cast<float>(g_jsEngine->toNumber(args[2])),
                    static_cast<float>(g_jsEngine->toNumber(args[3]))
                );
            }
            return g_jsEngine->newUndefined();
        })
    );

    // beginPath()
    engine->setProperty(jsCtx, "beginPath",
        engine->newFunction("beginPath", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx) capturedCtx->beginPath();
            return g_jsEngine->newUndefined();
        })
    );

    // closePath()
    engine->setProperty(jsCtx, "closePath",
        engine->newFunction("closePath", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx) capturedCtx->closePath();
            return g_jsEngine->newUndefined();
        })
    );

    // moveTo(x, y)
    engine->setProperty(jsCtx, "moveTo",
        engine->newFunction("moveTo", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 2) {
                capturedCtx->moveTo(
                    static_cast<float>(g_jsEngine->toNumber(args[0])),
                    static_cast<float>(g_jsEngine->toNumber(args[1]))
                );
            }
            return g_jsEngine->newUndefined();
        })
    );

    // lineTo(x, y)
    engine->setProperty(jsCtx, "lineTo",
        engine->newFunction("lineTo", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 2) {
                capturedCtx->lineTo(
                    static_cast<float>(g_jsEngine->toNumber(args[0])),
                    static_cast<float>(g_jsEngine->toNumber(args[1]))
                );
            }
            return g_jsEngine->newUndefined();
        })
    );

    // quadraticCurveTo(cpx, cpy, x, y)
    engine->setProperty(jsCtx, "quadraticCurveTo",
        engine->newFunction("quadraticCurveTo", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 4) {
                capturedCtx->quadraticCurveTo(
                    static_cast<float>(g_jsEngine->toNumber(args[0])),
                    static_cast<float>(g_jsEngine->toNumber(args[1])),
                    static_cast<float>(g_jsEngine->toNumber(args[2])),
                    static_cast<float>(g_jsEngine->toNumber(args[3]))
                );
            }
            return g_jsEngine->newUndefined();
        })
    );

    // bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x, y)
    engine->setProperty(jsCtx, "bezierCurveTo",
        engine->newFunction("bezierCurveTo", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 6) {
                capturedCtx->bezierCurveTo(
                    static_cast<float>(g_jsEngine->toNumber(args[0])),
                    static_cast<float>(g_jsEngine->toNumber(args[1])),
                    static_cast<float>(g_jsEngine->toNumber(args[2])),
                    static_cast<float>(g_jsEngine->toNumber(args[3])),
                    static_cast<float>(g_jsEngine->toNumber(args[4])),
                    static_cast<float>(g_jsEngine->toNumber(args[5]))
                );
            }
            return g_jsEngine->newUndefined();
        })
    );

    // arc(x, y, radius, startAngle, endAngle, counterclockwise)
    engine->setProperty(jsCtx, "arc",
        engine->newFunction("arc", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 5) {
                bool ccw = args.size() > 5 ? g_jsEngine->toBoolean(args[5]) : false;
                capturedCtx->arc(
                    static_cast<float>(g_jsEngine->toNumber(args[0])),
                    static_cast<float>(g_jsEngine->toNumber(args[1])),
                    static_cast<float>(g_jsEngine->toNumber(args[2])),
                    static_cast<float>(g_jsEngine->toNumber(args[3])),
                    static_cast<float>(g_jsEngine->toNumber(args[4])),
                    ccw
                );
            }
            return g_jsEngine->newUndefined();
        })
    );

    // fill()
    engine->setProperty(jsCtx, "fill",
        engine->newFunction("fill", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx) capturedCtx->fill();
            return g_jsEngine->newUndefined();
        })
    );

    // stroke()
    engine->setProperty(jsCtx, "stroke",
        engine->newFunction("stroke", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx) capturedCtx->stroke();
            return g_jsEngine->newUndefined();
        })
    );

    // getImageData(x, y, width, height) -> ImageData
    engine->setProperty(jsCtx, "getImageData",
        engine->newFunction("getImageData", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            auto result = g_jsEngine->newObject();
            if (capturedCtx && args.size() >= 4) {
                int x = static_cast<int>(g_jsEngine->toNumber(args[0]));
                int y = static_cast<int>(g_jsEngine->toNumber(args[1]));
                int w = static_cast<int>(g_jsEngine->toNumber(args[2]));
                int h = static_cast<int>(g_jsEngine->toNumber(args[3]));

                ImageData imgData = capturedCtx->getImageData(x, y, w, h);

                g_jsEngine->setProperty(result, "width", g_jsEngine->newNumber(imgData.width));
                g_jsEngine->setProperty(result, "height", g_jsEngine->newNumber(imgData.height));

                // Create Uint8Array for data (ImageData.data is Uint8ClampedArray in browsers)
                // Using Uint8Array allows direct indexing with []
                auto dataArray = g_jsEngine->createUint8Array(imgData.data.data(), imgData.data.size());
                g_jsEngine->setProperty(result, "data", dataArray);
            }
            return result;
        })
    );

    // putImageData(imageData, x, y)
    engine->setProperty(jsCtx, "putImageData",
        engine->newFunction("putImageData", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 3) {
                auto imageDataObj = args[0];
                int x = static_cast<int>(g_jsEngine->toNumber(args[1]));
                int y = static_cast<int>(g_jsEngine->toNumber(args[2]));

                // Extract ImageData properties
                int width = static_cast<int>(g_jsEngine->toNumber(g_jsEngine->getProperty(imageDataObj, "width")));
                int height = static_cast<int>(g_jsEngine->toNumber(g_jsEngine->getProperty(imageDataObj, "height")));
                auto dataHandle = g_jsEngine->getProperty(imageDataObj, "data");

                // Get the data array
                size_t dataSize = 0;
                void* dataPtr = g_jsEngine->getArrayBufferData(dataHandle, &dataSize);

                if (dataPtr && dataSize > 0) {
                    ImageData imgData;
                    imgData.width = width;
                    imgData.height = height;
                    imgData.data.assign(static_cast<uint8_t*>(dataPtr),
                                       static_cast<uint8_t*>(dataPtr) + dataSize);
                    capturedCtx->putImageData(imgData, x, y);
                }
            }
            return g_jsEngine->newUndefined();
        })
    );

    // createImageData(width, height) -> ImageData
    engine->setProperty(jsCtx, "createImageData",
        engine->newFunction("createImageData", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            auto result = g_jsEngine->newObject();
            if (args.size() >= 2) {
                int width = static_cast<int>(g_jsEngine->toNumber(args[0]));
                int height = static_cast<int>(g_jsEngine->toNumber(args[1]));

                g_jsEngine->setProperty(result, "width", g_jsEngine->newNumber(width));
                g_jsEngine->setProperty(result, "height", g_jsEngine->newNumber(height));

                // Create Uint8Array filled with zeros (transparent black)
                size_t dataSize = width * height * 4;
                std::vector<uint8_t> data(dataSize, 0);
                auto dataArray = g_jsEngine->createUint8Array(data.data(), data.size());
                g_jsEngine->setProperty(result, "data", dataArray);
            }
            return result;
        })
    );

    // drawImage - draws another canvas or image onto this canvas
    // Supports: drawImage(image, dx, dy)
    //           drawImage(image, dx, dy, dWidth, dHeight)
    //           drawImage(image, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight)
    engine->setProperty(jsCtx, "drawImage",
        engine->newFunction("drawImage", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (!capturedCtx || args.empty()) {
                return g_jsEngine->newUndefined();
            }

            auto imageArg = args[0];

            // Check if it's a canvas element (has getContext method and _context2d)
            auto context2d = g_jsEngine->getProperty(imageArg, "_context2d");
            if (!g_jsEngine->isUndefined(context2d) && !g_jsEngine->isNull(context2d)) {
                // It's a canvas element, get its Canvas2DContext
                Canvas2DContext* sourceCtx = static_cast<Canvas2DContext*>(g_jsEngine->getPrivateData(context2d));
                if (sourceCtx) {
                    // Get source canvas dimensions
                    int srcWidth = sourceCtx->getWidth();
                    int srcHeight = sourceCtx->getHeight();

                    // Get the pixel data from source canvas
                    ImageData srcData = sourceCtx->getImageData(0, 0, srcWidth, srcHeight);

                    if (args.size() == 3) {
                        // drawImage(image, dx, dy)
                        int dx = static_cast<int>(g_jsEngine->toNumber(args[1]));
                        int dy = static_cast<int>(g_jsEngine->toNumber(args[2]));
                        capturedCtx->putImageData(srcData, dx, dy);
                    } else if (args.size() == 5) {
                        // drawImage(image, dx, dy, dWidth, dHeight) - scaled
                        // For now, just use putImageData without scaling
                        // TODO: Implement scaling
                        int dx = static_cast<int>(g_jsEngine->toNumber(args[1]));
                        int dy = static_cast<int>(g_jsEngine->toNumber(args[2]));
                        capturedCtx->putImageData(srcData, dx, dy);
                    } else if (args.size() >= 9) {
                        // drawImage(image, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight)
                        int sx = static_cast<int>(g_jsEngine->toNumber(args[1]));
                        int sy = static_cast<int>(g_jsEngine->toNumber(args[2]));
                        int sWidth = static_cast<int>(g_jsEngine->toNumber(args[3]));
                        int sHeight = static_cast<int>(g_jsEngine->toNumber(args[4]));
                        int dx = static_cast<int>(g_jsEngine->toNumber(args[5]));
                        int dy = static_cast<int>(g_jsEngine->toNumber(args[6]));
                        // dWidth and dHeight for scaling (ignored for now)

                        // Get sub-region of source
                        ImageData subData = sourceCtx->getImageData(sx, sy, sWidth, sHeight);
                        capturedCtx->putImageData(subData, dx, dy);
                    }
                }
            }
            // TODO: Support HTMLImageElement and ImageBitmap

            return g_jsEngine->newUndefined();
        })
    );

    // Transform methods for PixiJS font rendering

    // scale(x, y)
    engine->setProperty(jsCtx, "scale",
        engine->newFunction("scale", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 2) {
                capturedCtx->scale(
                    static_cast<float>(g_jsEngine->toNumber(args[0])),
                    static_cast<float>(g_jsEngine->toNumber(args[1]))
                );
            }
            return g_jsEngine->newUndefined();
        })
    );

    // rotate(angle)
    engine->setProperty(jsCtx, "rotate",
        engine->newFunction("rotate", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 1) {
                capturedCtx->rotate(static_cast<float>(g_jsEngine->toNumber(args[0])));
            }
            return g_jsEngine->newUndefined();
        })
    );

    // translate(x, y)
    engine->setProperty(jsCtx, "translate",
        engine->newFunction("translate", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 2) {
                capturedCtx->translate(
                    static_cast<float>(g_jsEngine->toNumber(args[0])),
                    static_cast<float>(g_jsEngine->toNumber(args[1]))
                );
            }
            return g_jsEngine->newUndefined();
        })
    );

    // setTransform(a, b, c, d, e, f)
    engine->setProperty(jsCtx, "setTransform",
        engine->newFunction("setTransform", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 6) {
                capturedCtx->setTransform(
                    static_cast<float>(g_jsEngine->toNumber(args[0])),
                    static_cast<float>(g_jsEngine->toNumber(args[1])),
                    static_cast<float>(g_jsEngine->toNumber(args[2])),
                    static_cast<float>(g_jsEngine->toNumber(args[3])),
                    static_cast<float>(g_jsEngine->toNumber(args[4])),
                    static_cast<float>(g_jsEngine->toNumber(args[5]))
                );
            }
            return g_jsEngine->newUndefined();
        })
    );

    // transform(a, b, c, d, e, f)
    engine->setProperty(jsCtx, "transform",
        engine->newFunction("transform", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx && args.size() >= 6) {
                capturedCtx->transform(
                    static_cast<float>(g_jsEngine->toNumber(args[0])),
                    static_cast<float>(g_jsEngine->toNumber(args[1])),
                    static_cast<float>(g_jsEngine->toNumber(args[2])),
                    static_cast<float>(g_jsEngine->toNumber(args[3])),
                    static_cast<float>(g_jsEngine->toNumber(args[4])),
                    static_cast<float>(g_jsEngine->toNumber(args[5]))
                );
            }
            return g_jsEngine->newUndefined();
        })
    );

    // resetTransform()
    engine->setProperty(jsCtx, "resetTransform",
        engine->newFunction("resetTransform", [capturedCtx](void* c, const std::vector<js::JSValueHandle>& args) {
            if (capturedCtx) {
                capturedCtx->resetTransform();
            }
            return g_jsEngine->newUndefined();
        })
    );

    std::cout << "[Canvas2D] JS bindings created" << std::endl;
    return jsCtx;
}

/**
 * Create a new Canvas2D context for a canvas element
 *
 * This function creates both the native Canvas2DContext (Skia-backed) and
 * the JavaScript bindings. Each context captures its own native pointer in
 * closures, allowing multiple canvas contexts to work independently.
 *
 * @param engine The JS engine
 * @param width Canvas width
 * @param height Canvas height
 * @return JS object representing the CanvasRenderingContext2D
 */
js::JSValueHandle createCanvas2DContext(js::Engine* engine, int width, int height) {
    // Create native context
    auto nativeCtx = std::make_unique<Canvas2DContext>(width, height);
    Canvas2DContext* ctxPtr = nativeCtx.get();

    // Create JS bindings (methods capture ctxPtr in their closures)
    auto jsCtx = createCanvas2DJSObject(engine, ctxPtr);

    // Store the native context to prevent deletion
    g_canvas2dContexts[ctxPtr] = std::move(nativeCtx);

    // Protect the JS object from garbage collection
    engine->protect(jsCtx);

    // Add native setter methods that capture the context pointer
    // These are called by the property interceptors below
    engine->setProperty(jsCtx, "__nativeSetFillStyle",
        engine->newFunction("__nativeSetFillStyle", [ctxPtr](void* c, const std::vector<js::JSValueHandle>& args) {
            if (ctxPtr && !args.empty()) {
                ctxPtr->setFillStyle(g_jsEngine->toString(args[0]));
            }
            return g_jsEngine->newUndefined();
        })
    );

    engine->setProperty(jsCtx, "__nativeSetStrokeStyle",
        engine->newFunction("__nativeSetStrokeStyle", [ctxPtr](void* c, const std::vector<js::JSValueHandle>& args) {
            if (ctxPtr && !args.empty()) {
                ctxPtr->setStrokeStyle(g_jsEngine->toString(args[0]));
            }
            return g_jsEngine->newUndefined();
        })
    );

    engine->setProperty(jsCtx, "__nativeSetLineWidth",
        engine->newFunction("__nativeSetLineWidth", [ctxPtr](void* c, const std::vector<js::JSValueHandle>& args) {
            if (ctxPtr && !args.empty()) {
                ctxPtr->setLineWidth(static_cast<float>(g_jsEngine->toNumber(args[0])));
            }
            return g_jsEngine->newUndefined();
        })
    );

    engine->setProperty(jsCtx, "__nativeSetGlobalAlpha",
        engine->newFunction("__nativeSetGlobalAlpha", [ctxPtr](void* c, const std::vector<js::JSValueHandle>& args) {
            if (ctxPtr && !args.empty()) {
                ctxPtr->setGlobalAlpha(static_cast<float>(g_jsEngine->toNumber(args[0])));
            }
            return g_jsEngine->newUndefined();
        })
    );

    engine->setProperty(jsCtx, "__nativeSetFont",
        engine->newFunction("__nativeSetFont", [ctxPtr](void* c, const std::vector<js::JSValueHandle>& args) {
            if (ctxPtr && !args.empty()) {
                ctxPtr->setFont(g_jsEngine->toString(args[0]));
            }
            return g_jsEngine->newUndefined();
        })
    );

    engine->setProperty(jsCtx, "__nativeSetTextAlign",
        engine->newFunction("__nativeSetTextAlign", [ctxPtr](void* c, const std::vector<js::JSValueHandle>& args) {
            if (ctxPtr && !args.empty()) {
                ctxPtr->setTextAlign(g_jsEngine->toString(args[0]));
            }
            return g_jsEngine->newUndefined();
        })
    );

    engine->setProperty(jsCtx, "__nativeSetTextBaseline",
        engine->newFunction("__nativeSetTextBaseline", [ctxPtr](void* c, const std::vector<js::JSValueHandle>& args) {
            if (ctxPtr && !args.empty()) {
                ctxPtr->setTextBaseline(g_jsEngine->toString(args[0]));
            }
            return g_jsEngine->newUndefined();
        })
    );

    engine->setProperty(jsCtx, "__nativeResizeCanvas",
        engine->newFunction("__nativeResizeCanvas", [ctxPtr](void*, const std::vector<js::JSValueHandle>& args) {
            if (args.size() < 2) return g_jsEngine->newUndefined();
            const double width = g_jsEngine->toNumber(args[0]);
            const double height = g_jsEngine->toNumber(args[1]);
            if (!std::isfinite(width) || !std::isfinite(height) || width < 0 || height < 0 ||
                width > 65535 || height > 65535 || std::floor(width) != width || std::floor(height) != height) {
                g_jsEngine->throwException("Invalid Canvas 2D drawing-buffer dimensions");
                return g_jsEngine->newUndefined();
            }
            ctxPtr->resize(static_cast<int>(width), static_cast<int>(height));
            return g_jsEngine->newUndefined();
        })
    );

    // Store this context temporarily for the property interceptor setup
    // This is only needed for the eval() call below and is immediately overwritten
    // when another context is created, but that's fine because we only use it
    // inside the IIFE that runs synchronously
    engine->setGlobalProperty("__canvas2dContextTemp", jsCtx);

    // Set up property interceptors for fillStyle, strokeStyle, etc.
    // These call the native setters when properties are changed
    // The IIFE receives the context as a parameter (not via global lookup)
    const char* setupPropertyInterceptors = R"(
        (function(ctx) {
            var _fillStyle = '#000000';
            var _strokeStyle = '#000000';
            var _lineWidth = 1.0;
            var _globalAlpha = 1.0;
            var _font = '10px sans-serif';
            var _textAlign = 'start';
            var _textBaseline = 'alphabetic';
            var _states = [];
            var nativeSave = ctx.save;
            var nativeRestore = ctx.restore;
            ctx.save = function() {
                _states.push([_fillStyle, _strokeStyle, _lineWidth, _globalAlpha, _font, _textAlign, _textBaseline]);
                return nativeSave();
            };
            ctx.restore = function() {
                if (_states.length) {
                    [_fillStyle, _strokeStyle, _lineWidth, _globalAlpha, _font, _textAlign, _textBaseline] = _states.pop();
                }
                return nativeRestore();
            };
            ctx.__resizeDrawingBuffer = function(width, height) {
                ctx.__nativeResizeCanvas(width, height);
                _fillStyle = _strokeStyle = '#000000';
                _lineWidth = _globalAlpha = 1;
                _font = '10px sans-serif';
                _textAlign = 'start';
                _textBaseline = 'alphabetic';
                _states = [];
            };

            Object.defineProperty(ctx, 'fillStyle', {
                get: function() { return _fillStyle; },
                set: function(v) {
                    _fillStyle = v;
                    ctx.__nativeSetFillStyle(v);
                }
            });

            Object.defineProperty(ctx, 'strokeStyle', {
                get: function() { return _strokeStyle; },
                set: function(v) {
                    _strokeStyle = v;
                    ctx.__nativeSetStrokeStyle(v);
                }
            });

            Object.defineProperty(ctx, 'lineWidth', {
                get: function() { return _lineWidth; },
                set: function(v) {
                    _lineWidth = v;
                    ctx.__nativeSetLineWidth(v);
                }
            });

            Object.defineProperty(ctx, 'globalAlpha', {
                get: function() { return _globalAlpha; },
                set: function(v) {
                    _globalAlpha = v;
                    ctx.__nativeSetGlobalAlpha(v);
                }
            });

            Object.defineProperty(ctx, 'font', {
                get: function() { return _font; },
                set: function(v) {
                    _font = v;
                    ctx.__nativeSetFont(v);
                }
            });

            Object.defineProperty(ctx, 'textAlign', {
                get: function() { return _textAlign; },
                set: function(v) {
                    _textAlign = v;
                    ctx.__nativeSetTextAlign(v);
                }
            });

            Object.defineProperty(ctx, 'textBaseline', {
                get: function() { return _textBaseline; },
                set: function(v) {
                    _textBaseline = v;
                    ctx.__nativeSetTextBaseline(v);
                }
            });
        })(__canvas2dContextTemp);
    )";

    // Execute the property interceptor setup
    engine->eval(setupPropertyInterceptors, "canvas2d-setup");

    return jsCtx;
}

/**
 * Get the native Canvas2DContext from a JS context object
 */
Canvas2DContext* getCanvas2DContextFromJS(js::Engine* engine, js::JSValueHandle jsCtx) {
    return static_cast<Canvas2DContext*>(engine->getPrivateData(jsCtx));
}

}  // namespace canvas
}  // namespace mystral
