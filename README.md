# 2D SDF Renderer

Experimental 2D Rendering engine based on signed distance functions (SDFs).

<image src="./image.png" width="500">

---

## Features
- Basic shape rendering (Line, Rect, RoundedRect, Circle, Polygon)
- Mediocre Frametimes
- Anti-Aliasing
- Inflation and deflation
- Shadows (for some obscure reason only or one object, or to be specific the last element from the object descriptor table)

### Planned Features
- More shapes (BezierCurves, B-Splines)
- Translation, Rotation, Scale
- Text Rendering
- Faster approximate mode (calculation of SDFs only in bounding box regions, etc.)
- Outlines

---

## License

This project ist licensed under the MIT License.