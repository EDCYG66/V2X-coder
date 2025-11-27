from typing import List, Dict, Tuple, Optional
import math

def generate_four_direction_dual_lanes(
    width: float = 800,
    height: float = 800,
    lane_width: float = 3.5,
    lanes_per_dir: int = 2,
    median_gap_factor: float = 0.8,
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
    bs_dx: float = 0.0,
    bs_dy: float = 0.0,
) -> Dict:
    """
    生成四向十字交叉口（每向 lanes_per_dir 条直行车道）和可偏移基站位置。
    返回:
      - up_lanes, down_lanes, left_lanes, right_lanes: 车道中心线坐标
      - width, height
      - bs_position: (bs_x, bs_y)
      - lane_meta: 详细车道描述列表 (direction, order, coord)
    参数:
      median_gap_factor * lane_width 作为双向车道组之间的中隔
      bs_dx, bs_dy: 基站相对几何中心的偏移 (米，右/上为正)
    """
    if center_x is None:
        center_x = width / 2
    if center_y is None:
        center_y = height / 2

    median_gap = median_gap_factor * lane_width

    up_lanes, down_lanes = [], []
    # 上(+Y)在几何中心左侧排列；下(-Y)在右侧
    for k in range(lanes_per_dir):
        offset = (k + 0.5) * lane_width
        up_lanes.append(center_x - median_gap / 2 - offset)
        down_lanes.append(center_x + median_gap / 2 + offset)

    right_lanes, left_lanes = [], []
    # 右(+X)在中心下侧；左(-X)在中心上侧
    for k in range(lanes_per_dir):
        offset = (k + 0.5) * lane_width
        right_lanes.append(center_y - median_gap / 2 - offset)
        left_lanes.append(center_y + median_gap / 2 + offset)

    bs_x = center_x + bs_dx
    bs_y = center_y + bs_dy

    lane_meta = []
    # 统一登记，便于输出/可视化
    for idx, x in enumerate(up_lanes):
        lane_meta.append(dict(direction='u', order=idx, axis='x', coord=x))
    for idx, x in enumerate(down_lanes):
        lane_meta.append(dict(direction='d', order=idx, axis='x', coord=x))
    for idx, y in enumerate(left_lanes):
        lane_meta.append(dict(direction='l', order=idx, axis='y', coord=y))
    for idx, y in enumerate(right_lanes):
        lane_meta.append(dict(direction='r', order=idx, axis='y', coord=y))

    return dict(
        up_lanes=up_lanes,
        down_lanes=down_lanes,
        left_lanes=left_lanes,
        right_lanes=right_lanes,
        width=width,
        height=height,
        bs_position=(bs_x, bs_y),
        center=(center_x, center_y),
        lane_width=lane_width,
        median_gap=median_gap,
        lanes_per_dir=lanes_per_dir,
        lane_meta=lane_meta
    )

def pretty_print_lane_meta(lane_meta: List[Dict]):
    """
    便于在控制台查看生成的车道。
    """
    print("Lane Detail (direction u=up +Y, d=down -Y, r=right +X, l=left -X)")
    for item in lane_meta:
        print(f"  dir={item['direction']} order={item['order']} axis={item['axis']} coord={item['coord']:.2f}")

if __name__ == "__main__":
    layout = generate_four_direction_dual_lanes(bs_dx=40, bs_dy=-30)
    pretty_print_lane_meta(layout['lane_meta'])
    print("BS at:", layout['bs_position'])