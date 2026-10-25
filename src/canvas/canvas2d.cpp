/**
 * Canvas 2D Context Implementation
 *
 * Uses Skia for hardware-accelerated 2D graphics when available.
 * Falls back to a stub implementation when Skia is not present.
 */

#include "mystral/canvas/canvas2d.h"
#include <iostream>
#include <cmath>
#include <regex>
#include <stack>

// M_PI is not defined by default on Windows MSVC
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(MYSTRAL_HAS_SKIA)
// Skia headers (include path is third_party/skia/build/include, headers use "include/core/..." internally)
#include "include/core/SkCanvas.h"
#include "include/core/SkSurface.h"
#include "include/core/SkPath.h"
#include "include/core/SkPathBuilder.h"  // Skia m145+ uses SkPathBuilder for path construction
#include "include/core/SkPaint.h"
#include "include/core/SkFont.h"
#include "include/core/SkFontMgr.h"
#include "include/core/SkFontMetrics.h"
#include "include/core/SkTypeface.h"
#include "include/core/SkColor.h"
#include "include/core/SkColorSpace.h"
#include "include/core/SkImageInfo.h"
#include "include/core/SkBitmap.h"
#include "include/core/SkPixmap.h"
#include "include/core/SkImage.h"

// Platform-specific font manager
#if defined(__APPLE__)
#include "include/ports/SkFontMgr_mac_ct.h"
#elif defined(_WIN32)
#include "include/ports/SkTypeface_win.h"
#elif defined(__ANDROID__)
#include "include/ports/SkFontMgr_android.h"
#include "include/ports/SkFontScanner_FreeType.h"
#else
#include "include/ports/SkFontMgr_fontconfig.h"
#endif
#endif

namespace mystral {
namespace canvas {

// ============================================================================
// Color Parsing Utilities
// ============================================================================

struct Color {
    uint8_t r = 0, g = 0, b = 0, a = 255;
};

static Color parseColor(const std::string& colorStr) {
    Color color;

    if (colorStr.empty()) {
        return color;
    }

    // Handle hex colors: #RGB, #RGBA, #RRGGBB, #RRGGBBAA
    if (colorStr[0] == '#') {
        std::string hex = colorStr.substr(1);
        if (hex.length() == 3) {
            // #RGB -> #RRGGBB
            color.r = std::stoi(std::string(2, hex[0]), nullptr, 16);
            color.g = std::stoi(std::string(2, hex[1]), nullptr, 16);
            color.b = std::stoi(std::string(2, hex[2]), nullptr, 16);
        } else if (hex.length() == 4) {
            // #RGBA -> #RRGGBBAA
            color.r = std::stoi(std::string(2, hex[0]), nullptr, 16);
            color.g = std::stoi(std::string(2, hex[1]), nullptr, 16);
            color.b = std::stoi(std::string(2, hex[2]), nullptr, 16);
            color.a = std::stoi(std::string(2, hex[3]), nullptr, 16);
        } else if (hex.length() == 6) {
            color.r = std::stoi(hex.substr(0, 2), nullptr, 16);
            color.g = std::stoi(hex.substr(2, 2), nullptr, 16);
            color.b = std::stoi(hex.substr(4, 2), nullptr, 16);
        } else if (hex.length() == 8) {
            color.r = std::stoi(hex.substr(0, 2), nullptr, 16);
            color.g = std::stoi(hex.substr(2, 2), nullptr, 16);
            color.b = std::stoi(hex.substr(4, 2), nullptr, 16);
            color.a = std::stoi(hex.substr(6, 2), nullptr, 16);
        }
        return color;
    }

    // Handle rgb(r, g, b) and rgba(r, g, b, a)
    std::regex rgbaRegex(R"(rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+))?\s*\))");
    std::smatch match;
    if (std::regex_match(colorStr, match, rgbaRegex)) {
        color.r = std::stoi(match[1]);
        color.g = std::stoi(match[2]);
        color.b = std::stoi(match[3]);
        if (match[4].matched) {
            float alpha = std::stof(match[4]);
            color.a = static_cast<uint8_t>(alpha * 255);
        }
        return color;
    }

    // Handle hsl(h, s%, l%) and hsla(h, s%, l%, a)
    std::regex hslRegex(R"(hsla?\s*\(\s*([\d.]+)\s*,\s*([\d.]+)%?\s*,\s*([\d.]+)%?\s*(?:,\s*([\d.]+))?\s*\))");
    if (std::regex_match(colorStr, match, hslRegex)) {
        float h = std::fmod(std::stof(match[1]), 360.0f);
        if (h < 0) h += 360.0f;
        float s = std::stof(match[2]) / 100.0f;
        float l = std::stof(match[3]) / 100.0f;
        float a = match[4].matched ? std::stof(match[4]) : 1.0f;

        // HSL to RGB conversion
        float c = (1.0f - std::fabs(2.0f * l - 1.0f)) * s;
        float x = c * (1.0f - std::fabs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
        float m = l - c / 2.0f;

        float r1, g1, b1;
        if (h < 60)       { r1 = c; g1 = x; b1 = 0; }
        else if (h < 120) { r1 = x; g1 = c; b1 = 0; }
        else if (h < 180) { r1 = 0; g1 = c; b1 = x; }
        else if (h < 240) { r1 = 0; g1 = x; b1 = c; }
        else if (h < 300) { r1 = x; g1 = 0; b1 = c; }
        else              { r1 = c; g1 = 0; b1 = x; }

        color.r = static_cast<uint8_t>((r1 + m) * 255);
        color.g = static_cast<uint8_t>((g1 + m) * 255);
        color.b = static_cast<uint8_t>((b1 + m) * 255);
        color.a = static_cast<uint8_t>(a * 255);
        return color;
    }

    // Named colors (basic set)
    if (colorStr == "black") { color.r = 0; color.g = 0; color.b = 0; }
    else if (colorStr == "white") { color.r = 255; color.g = 255; color.b = 255; }
    else if (colorStr == "red") { color.r = 255; color.g = 0; color.b = 0; }
    else if (colorStr == "green") { color.r = 0; color.g = 128; color.b = 0; }
    else if (colorStr == "blue") { color.r = 0; color.g = 0; color.b = 255; }
    else if (colorStr == "transparent") { color.r = 0; color.g = 0; color.b = 0; color.a = 0; }

    return color;
}

// ============================================================================
// Font Parsing Utilities
// ============================================================================

struct FontInfo {
    float size = 16.0f;
    std::string family = "sans-serif";
    bool bold = false;
    bool italic = false;
};

static FontInfo parseFont(const std::string& fontStr) {
    FontInfo info;

    // Parse CSS font string: "italic bold 16px Arial"
    // Simplified parser - handles: [style] [weight] size[px/pt] family
    std::regex fontRegex(R"((?:(italic|oblique)\s+)?(?:(bold|normal|\d+)\s+)?(\d+(?:\.\d+)?)(px|pt|em)\s+(.+))");
    std::smatch match;

    if (std::regex_match(fontStr, match, fontRegex)) {
        if (match[1].matched) {
            info.italic = (match[1] == "italic" || match[1] == "oblique");
        }
        if (match[2].matched) {
            std::string weight = match[2];
            info.bold = (weight == "bold" || (weight != "normal" && std::stoi(weight) >= 700));
        }
        info.size = std::stof(match[3]);
        std::string unit = match[4];
        if (unit == "pt") {
            info.size *= 1.333f;  // pt to px conversion
        } else if (unit == "em") {
            info.size *= 16.0f;  // Assume 16px base
        }
        info.family = match[5];
    } else {
        // Fallback: just try to extract size
        std::regex sizeRegex(R"((\d+(?:\.\d+)?)(px|pt))");
        if (std::regex_search(fontStr, match, sizeRegex)) {
            info.size = std::stof(match[1]);
        }
    }

    return info;
}

// ============================================================================
// Canvas2D State (for save/restore)
// ============================================================================

struct Canvas2DState {
    std::string fillStyle = "#000000";
    std::string strokeStyle = "#000000";
    float lineWidth = 1.0f;
    float globalAlpha = 1.0f;
    std::string font = "10px sans-serif";
    std::string textAlign = "start";
    std::string textBaseline = "alphabetic";
};

// ============================================================================
// Implementation
// ============================================================================

#if defined(MYSTRAL_HAS_SKIA)

struct Canvas2DContext::Impl {
    sk_sp<SkSurface> surface;
    SkCanvas* canvas = nullptr;  // Owned by surface
    SkPathBuilder pathBuilder;  // Skia m145+ uses SkPathBuilder for path construction
    Canvas2DState currentState;
    std::stack<Canvas2DState> stateStack;
    sk_sp<SkFontMgr> fontMgr;
    sk_sp<SkTypeface> currentTypeface;
    SkFont currentFont;

    Impl(int width, int height) {
        // Create RGBA surface using new API (SkSurfaces namespace)
        SkImageInfo info = SkImageInfo::Make(
            width, height,
            kRGBA_8888_SkColorType,
            kPremul_SkAlphaType
        );
        surface = SkSurfaces::Raster(info);
        if (surface) {
            canvas = surface->getCanvas();
            canvas->clear(SK_ColorTRANSPARENT);
        }

        // Initialize font manager (platform-specific)
#if defined(__APPLE__)
        fontMgr = SkFontMgr_New_CoreText(nullptr);
#elif defined(_WIN32)
        fontMgr = SkFontMgr_New_DirectWrite();
#elif defined(__ANDROID__)
        fontMgr = SkFontMgr_New_Android(nullptr, SkFontScanner_Make_FreeType());
#else
        fontMgr = SkFontMgr::RefEmpty();  // Fallback
#endif
        if (fontMgr) {
            currentTypeface = fontMgr->matchFamilyStyle("sans-serif", SkFontStyle::Normal());
            if (!currentTypeface) {
                currentTypeface = fontMgr->matchFamilyStyle(nullptr, SkFontStyle::Normal());
            }
        }
        currentFont = SkFont(currentTypeface, 10.0f);
    }

    void resize(int width, int height) {
        SkImageInfo info = SkImageInfo::Make(
            width, height,
            kRGBA_8888_SkColorType,
            kPremul_SkAlphaType
        );
        surface = SkSurfaces::Raster(info);
        canvas = surface ? surface->getCanvas() : nullptr;
        if (canvas) canvas->clear(SK_ColorTRANSPARENT);
        currentState = Canvas2DState{};
        stateStack = {};
        pathBuilder = SkPathBuilder();
        updateFont();
    }

    SkPaint makeFillPaint() {
        SkPaint paint;
        paint.setAntiAlias(true);
        paint.setStyle(SkPaint::kFill_Style);
        Color c = parseColor(currentState.fillStyle);
        paint.setColor(SkColorSetARGB(
            static_cast<uint8_t>(c.a * currentState.globalAlpha),
            c.r, c.g, c.b
        ));
        return paint;
    }

    SkPaint makeStrokePaint() {
        SkPaint paint;
        paint.setAntiAlias(true);
        paint.setStyle(SkPaint::kStroke_Style);
        paint.setStrokeWidth(currentState.lineWidth);
        Color c = parseColor(currentState.strokeStyle);
        paint.setColor(SkColorSetARGB(
            static_cast<uint8_t>(c.a * currentState.globalAlpha),
            c.r, c.g, c.b
        ));
        return paint;
    }

    void updateFont() {
        FontInfo fi = parseFont(currentState.font);
        SkFontStyle style = SkFontStyle(
            fi.bold ? SkFontStyle::kBold_Weight : SkFontStyle::kNormal_Weight,
            SkFontStyle::kNormal_Width,
            fi.italic ? SkFontStyle::kItalic_Slant : SkFontStyle::kUpright_Slant
        );

        if (fontMgr) {
            currentTypeface = fontMgr->matchFamilyStyle(fi.family.c_str(), style);
            if (!currentTypeface) {
                currentTypeface = fontMgr->matchFamilyStyle("sans-serif", style);
            }
            if (!currentTypeface) {
                currentTypeface = fontMgr->matchFamilyStyle(nullptr, style);
            }
        }

        currentFont = SkFont(currentTypeface, fi.size);
        currentFont.setEdging(SkFont::Edging::kSubpixelAntiAlias);
    }
};

#else  // No Skia - stub implementation

struct Canvas2DContext::Impl {
    Canvas2DState currentState;
    std::stack<Canvas2DState> stateStack;
    std::vector<uint8_t> pixelData;
    int pixelWidth = 0;
    int pixelHeight = 0;

    Impl(int width, int height) : pixelWidth(width), pixelHeight(height) {
        pixelData.resize(width * height * 4, 0);
        std::cout << "[Canvas2D] Stub implementation (no Skia)" << std::endl;
    }

    void resize(int width, int height) {
        pixelWidth = width;
        pixelHeight = height;
        pixelData.assign(width * height * 4, 0);
        currentState = Canvas2DState{};
        stateStack = {};
    }
};

#endif  // MYSTRAL_HAS_SKIA

// ============================================================================
// Canvas2DContext Public API
// ============================================================================

Canvas2DContext::Canvas2DContext(int width, int height)
    : width_(width), height_(height) {
    impl_ = std::make_unique<Impl>(width, height);
    std::cout << "[Canvas2D] Created " << width << "x" << height << " context" << std::endl;
}

Canvas2DContext::~Canvas2DContext() = default;

void Canvas2DContext::resize(int width, int height) {
    width_ = width;
    height_ = height;
    impl_->resize(width, height);
}

// State Management
void Canvas2DContext::save() {
    impl_->stateStack.push(impl_->currentState);
#if defined(MYSTRAL_HAS_SKIA)
    if (impl_->canvas) {
        impl_->canvas->save();  // Save Skia canvas transform state
    }
#endif
}

void Canvas2DContext::restore() {
    if (!impl_->stateStack.empty()) {
        impl_->currentState = impl_->stateStack.top();
        impl_->stateStack.pop();
#if defined(MYSTRAL_HAS_SKIA)
        if (impl_->canvas) {
            impl_->canvas->restore();  // Restore Skia canvas transform state
        }
        impl_->updateFont();
#endif
    }
}

// Fill and Stroke Styles
void Canvas2DContext::setFillStyle(const std::string& color) {
    impl_->currentState.fillStyle = color;
}

void Canvas2DContext::setStrokeStyle(const std::string& color) {
    impl_->currentState.strokeStyle = color;
}

void Canvas2DContext::setLineWidth(float width) {
    impl_->currentState.lineWidth = width;
}

void Canvas2DContext::setGlobalAlpha(float alpha) {
    impl_->currentState.globalAlpha = alpha;
}

std::string Canvas2DContext::getFillStyle() const {
    return impl_->currentState.fillStyle;
}

std::string Canvas2DContext::getStrokeStyle() const {
    return impl_->currentState.strokeStyle;
}

float Canvas2DContext::getLineWidth() const {
    return impl_->currentState.lineWidth;
}

float Canvas2DContext::getGlobalAlpha() const {
    return impl_->currentState.globalAlpha;
}

// Text
void Canvas2DContext::setFont(const std::string& font) {
    impl_->currentState.font = font;
#if defined(MYSTRAL_HAS_SKIA)
    impl_->updateFont();
#endif
}

void Canvas2DContext::setTextAlign(const std::string& align) {
    impl_->currentState.textAlign = align;
}

void Canvas2DContext::setTextBaseline(const std::string& baseline) {
    impl_->currentState.textBaseline = baseline;
}

std::string Canvas2DContext::getFont() const {
    return impl_->currentState.font;
}

std::string Canvas2DContext::getTextAlign() const {
    return impl_->currentState.textAlign;
}

std::string Canvas2DContext::getTextBaseline() const {
    return impl_->currentState.textBaseline;
}

void Canvas2DContext::fillText(const std::string& text, float x, float y) {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->canvas) return;

    SkPaint paint = impl_->makeFillPaint();

    // Adjust x based on textAlign
    SkScalar textWidth = impl_->currentFont.measureText(text.c_str(), text.length(), SkTextEncoding::kUTF8);
    if (impl_->currentState.textAlign == "center") {
        x -= textWidth / 2;
    } else if (impl_->currentState.textAlign == "right" || impl_->currentState.textAlign == "end") {
        x -= textWidth;
    }

    // Adjust y based on textBaseline
    SkFontMetrics metrics;
    impl_->currentFont.getMetrics(&metrics);
    if (impl_->currentState.textBaseline == "top") {
        y -= metrics.fAscent;
    } else if (impl_->currentState.textBaseline == "middle") {
        y -= (metrics.fAscent + metrics.fDescent) / 2;
    } else if (impl_->currentState.textBaseline == "bottom") {
        y -= metrics.fDescent;
    }
    // "alphabetic" is the default - no adjustment needed

    impl_->canvas->drawString(text.c_str(), x, y, impl_->currentFont, paint);
#endif
}

void Canvas2DContext::strokeText(const std::string& text, float x, float y) {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->canvas) return;

    SkPaint paint = impl_->makeStrokePaint();

    // Adjust x based on textAlign
    SkScalar textWidth = impl_->currentFont.measureText(text.c_str(), text.length(), SkTextEncoding::kUTF8);
    if (impl_->currentState.textAlign == "center") {
        x -= textWidth / 2;
    } else if (impl_->currentState.textAlign == "right" || impl_->currentState.textAlign == "end") {
        x -= textWidth;
    }

    // Adjust y based on textBaseline
    SkFontMetrics metrics;
    impl_->currentFont.getMetrics(&metrics);
    if (impl_->currentState.textBaseline == "top") {
        y -= metrics.fAscent;
    } else if (impl_->currentState.textBaseline == "middle") {
        y -= (metrics.fAscent + metrics.fDescent) / 2;
    } else if (impl_->currentState.textBaseline == "bottom") {
        y -= metrics.fDescent;
    }
    // "alphabetic" is the default - no adjustment needed

    impl_->canvas->drawString(text.c_str(), x, y, impl_->currentFont, paint);
#endif
}

TextMetrics Canvas2DContext::measureText(const std::string& text) {
    TextMetrics metrics;

#if defined(MYSTRAL_HAS_SKIA)
    SkRect bounds;
    metrics.width = impl_->currentFont.measureText(text.c_str(), text.length(), SkTextEncoding::kUTF8, &bounds);

    SkFontMetrics fm;
    impl_->currentFont.getMetrics(&fm);

    metrics.actualBoundingBoxLeft = -bounds.fLeft;
    metrics.actualBoundingBoxRight = bounds.fRight;
    metrics.actualBoundingBoxAscent = -bounds.fTop;
    metrics.actualBoundingBoxDescent = bounds.fBottom;
    metrics.fontBoundingBoxAscent = -fm.fAscent;
    metrics.fontBoundingBoxDescent = fm.fDescent;
#else
    // Stub: estimate width based on font size
    FontInfo fi = parseFont(impl_->currentState.font);
    metrics.width = text.length() * fi.size * 0.6f;  // Rough estimate
    metrics.fontBoundingBoxAscent = fi.size * 0.8f;
    metrics.fontBoundingBoxDescent = fi.size * 0.2f;
#endif

    return metrics;
}

// Rectangles
void Canvas2DContext::fillRect(float x, float y, float width, float height) {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->canvas) return;
    impl_->canvas->drawRect(SkRect::MakeXYWH(x, y, width, height), impl_->makeFillPaint());
#endif
}

void Canvas2DContext::strokeRect(float x, float y, float width, float height) {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->canvas) return;
    impl_->canvas->drawRect(SkRect::MakeXYWH(x, y, width, height), impl_->makeStrokePaint());
#endif
}

void Canvas2DContext::clearRect(float x, float y, float width, float height) {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->canvas) return;
    SkPaint clearPaint;
    clearPaint.setBlendMode(SkBlendMode::kClear);
    impl_->canvas->drawRect(SkRect::MakeXYWH(x, y, width, height), clearPaint);
#else
    // Stub: clear pixel data
    int x0 = std::max(0, static_cast<int>(x));
    int y0 = std::max(0, static_cast<int>(y));
    int x1 = std::min(impl_->pixelWidth, static_cast<int>(x + width));
    int y1 = std::min(impl_->pixelHeight, static_cast<int>(y + height));
    for (int py = y0; py < y1; py++) {
        for (int px = x0; px < x1; px++) {
            int idx = (py * impl_->pixelWidth + px) * 4;
            impl_->pixelData[idx] = 0;
            impl_->pixelData[idx + 1] = 0;
            impl_->pixelData[idx + 2] = 0;
            impl_->pixelData[idx + 3] = 0;
        }
    }
#endif
}

// Paths
void Canvas2DContext::beginPath() {
#if defined(MYSTRAL_HAS_SKIA)
    impl_->pathBuilder = SkPathBuilder();  // Reset path builder
#endif
}

void Canvas2DContext::closePath() {
#if defined(MYSTRAL_HAS_SKIA)
    impl_->pathBuilder.close();
#endif
}

void Canvas2DContext::moveTo(float x, float y) {
#if defined(MYSTRAL_HAS_SKIA)
    impl_->pathBuilder.moveTo(x, y);
#endif
}

void Canvas2DContext::lineTo(float x, float y) {
#if defined(MYSTRAL_HAS_SKIA)
    impl_->pathBuilder.lineTo(x, y);
#endif
}

void Canvas2DContext::quadraticCurveTo(float cpx, float cpy, float x, float y) {
#if defined(MYSTRAL_HAS_SKIA)
    impl_->pathBuilder.quadTo(cpx, cpy, x, y);
#endif
}

void Canvas2DContext::bezierCurveTo(float cp1x, float cp1y, float cp2x, float cp2y, float x, float y) {
#if defined(MYSTRAL_HAS_SKIA)
    impl_->pathBuilder.cubicTo(cp1x, cp1y, cp2x, cp2y, x, y);
#endif
}

void Canvas2DContext::arc(float x, float y, float radius, float startAngle, float endAngle, bool counterclockwise) {
#if defined(MYSTRAL_HAS_SKIA)
    if (radius <= 0) return;

    SkRect oval = SkRect::MakeLTRB(x - radius, y - radius, x + radius, y + radius);

    // Convert radians to degrees
    float startDeg = startAngle * 180.0f / M_PI;
    float sweepDeg = (endAngle - startAngle) * 180.0f / M_PI;

    if (counterclockwise && sweepDeg > 0) {
        sweepDeg -= 360.0f;
    } else if (!counterclockwise && sweepDeg < 0) {
        sweepDeg += 360.0f;
    }

    // For a full circle (or near-full), use addOval for better results
    if (std::fabs(sweepDeg) >= 359.9f) {
        impl_->pathBuilder.addOval(oval);
    } else {
        // For partial arcs, compute the start point and move there first
        float startRad = startAngle;
        float startX = x + radius * std::cos(startRad);
        float startY = y + radius * std::sin(startRad);

        // If path is empty, move to start point; otherwise line to it
        SkPath currentPath = impl_->pathBuilder.snapshot();
        if (currentPath.isEmpty()) {
            impl_->pathBuilder.moveTo(startX, startY);
        } else {
            impl_->pathBuilder.lineTo(startX, startY);
        }

        impl_->pathBuilder.arcTo(oval, startDeg, sweepDeg, false);
    }
#endif
}

void Canvas2DContext::arcTo(float x1, float y1, float x2, float y2, float radius) {
#if defined(MYSTRAL_HAS_SKIA)
    impl_->pathBuilder.arcTo(SkPoint::Make(x1, y1), SkPoint::Make(x2, y2), radius);
#endif
}

void Canvas2DContext::rect(float x, float y, float width, float height) {
#if defined(MYSTRAL_HAS_SKIA)
    impl_->pathBuilder.addRect(SkRect::MakeXYWH(x, y, width, height));
#endif
}

void Canvas2DContext::fill() {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->canvas) return;
    // snapshot() returns an SkPath without consuming the builder
    impl_->canvas->drawPath(impl_->pathBuilder.snapshot(), impl_->makeFillPaint());
#endif
}

void Canvas2DContext::stroke() {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->canvas) return;
    // snapshot() returns an SkPath without consuming the builder
    impl_->canvas->drawPath(impl_->pathBuilder.snapshot(), impl_->makeStrokePaint());
#endif
}

// Transformations
void Canvas2DContext::scale(float x, float y) {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->canvas) return;
    impl_->canvas->scale(x, y);
#endif
}

void Canvas2DContext::rotate(float angle) {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->canvas) return;
    impl_->canvas->rotate(SkRadiansToDegrees(angle));
#endif
}

void Canvas2DContext::translate(float x, float y) {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->canvas) return;
    impl_->canvas->translate(x, y);
#endif
}

void Canvas2DContext::setTransform(float a, float b, float c, float d, float e, float f) {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->canvas) return;
    // Canvas 2D transform matrix: [a c e]
    //                             [b d f]
    //                             [0 0 1]
    // SkMatrix uses column-major format
    SkMatrix matrix;
    matrix.setAll(a, c, e, b, d, f, 0, 0, 1);
    impl_->canvas->setMatrix(matrix);
#endif
}

void Canvas2DContext::transform(float a, float b, float c, float d, float e, float f) {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->canvas) return;
    SkMatrix matrix;
    matrix.setAll(a, c, e, b, d, f, 0, 0, 1);
    impl_->canvas->concat(matrix);
#endif
}

void Canvas2DContext::resetTransform() {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->canvas) return;
    impl_->canvas->resetMatrix();
#endif
}

// Pixel Manipulation
ImageData Canvas2DContext::getImageData(int x, int y, int width, int height) {
    ImageData data;
    data.width = width;
    data.height = height;
    data.data.resize(width * height * 4);

#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->surface) return data;

    SkImageInfo info = SkImageInfo::Make(width, height, kRGBA_8888_SkColorType, kUnpremul_SkAlphaType);
    impl_->surface->readPixels(info, data.data.data(), width * 4, x, y);
#else
    // Stub: return pixel data from our buffer
    for (int py = 0; py < height; py++) {
        for (int px = 0; px < width; px++) {
            int srcX = x + px;
            int srcY = y + py;
            if (srcX >= 0 && srcX < impl_->pixelWidth && srcY >= 0 && srcY < impl_->pixelHeight) {
                int srcIdx = (srcY * impl_->pixelWidth + srcX) * 4;
                int dstIdx = (py * width + px) * 4;
                data.data[dstIdx] = impl_->pixelData[srcIdx];
                data.data[dstIdx + 1] = impl_->pixelData[srcIdx + 1];
                data.data[dstIdx + 2] = impl_->pixelData[srcIdx + 2];
                data.data[dstIdx + 3] = impl_->pixelData[srcIdx + 3];
            }
        }
    }
#endif

    return data;
}

void Canvas2DContext::putImageData(const ImageData& imageData, int x, int y) {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->canvas) return;

    SkImageInfo info = SkImageInfo::Make(imageData.width, imageData.height, kRGBA_8888_SkColorType, kUnpremul_SkAlphaType);
    SkBitmap bitmap;
    bitmap.installPixels(info, const_cast<uint8_t*>(imageData.data.data()), imageData.width * 4);

    impl_->canvas->drawImage(bitmap.asImage(), x, y);
#endif
}

const uint8_t* Canvas2DContext::getPixelData() const {
#if defined(MYSTRAL_HAS_SKIA)
    if (!impl_->surface) return nullptr;

    // For raster surfaces, we can directly peek at the pixels
    SkPixmap pixmap;
    if (impl_->surface->peekPixels(&pixmap)) {
        return static_cast<const uint8_t*>(pixmap.addr());
    }
    return nullptr;
#else
    return impl_->pixelData.data();
#endif
}

size_t Canvas2DContext::getPixelDataSize() const {
    return width_ * height_ * 4;
}

}  // namespace canvas
}  // namespace mystral
