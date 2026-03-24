import cv2
import numpy as np

def draw_graph_cv2(canvas, data_x, data_y, color, rect, title="", line_thickness=1, y_min_fixed=None, y_max_fixed=None):
    """
    Draws a line chart on a given OpenCV canvas within a specific rectangle.

    Args:
        canvas: The NumPy array image to draw on.
        data_x: X-coordinates of the data (can be list or array).
        data_y: Y-coordinates of the data (can be list or array).
        color: Tuple (B, G, R) for the line color.
        rect: Tuple (x, y, w, h) defining the bounding box of the graph.
        title: Title string of the graph.
        line_thickness: Thickness of the plot line.
        y_min_fixed: Optional fixed minimum for the Y-axis.
        y_max_fixed: Optional fixed maximum for the Y-axis.
    """
    if data_y is None or len(data_y) < 2:
        return
    
    if data_x is None or len(data_x) != len(data_y):
        data_x = np.arange(len(data_y))

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    
    # Draw background and border
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # Put title
    cv2.putText(canvas, title, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Determine axis limits
    min_x = np.min(data_x)
    max_x = np.max(data_x)
    
    min_y = np.min(data_y) if y_min_fixed is None else y_min_fixed
    max_y = np.max(data_y) if y_max_fixed is None else y_max_fixed

    if max_x == min_x: max_x = min_x + 1
    if max_y == min_y:
        max_y = min_y + 1
        min_y = min_y - 1

    # Apply padding to y
    y_range = max_y - min_y
    if y_min_fixed is None: min_y -= y_range * 0.1
    if y_max_fixed is None: max_y += y_range * 0.1
    y_range = max_y - min_y

    padding = 25 # padding inside the rect
    plot_w = w - padding * 2
    plot_h = h - padding * 2
    plot_x_start = x + padding
    plot_y_end = y + h - padding

    # Transform coordinates
    def transform_pt(px, py):
        nx = plot_x_start + int(((px - min_x) / (max_x - min_x)) * plot_w)
        ny = plot_y_end - int(((py - min_y) / y_range) * plot_h)
        return (nx, ny)

    points = [transform_pt(px, py) for px, py in zip(data_x, data_y)]
    for i in range(1, len(points)):
        pt1 = points[i - 1]
        pt2 = points[i]
        cv2.line(canvas, pt1, pt2, color, line_thickness, cv2.LINE_AA)

    # Draw max/min labels
    cv2.putText(canvas, f"{max_y:.2f}", (x + 2, y + padding), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(canvas, f"{min_y:.2f}", (x + 2, plot_y_end), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
