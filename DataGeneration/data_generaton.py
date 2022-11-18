import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from TrajectryGeneration.models import Robot_model
from TrajectryGeneration.objects.field import Field, Rectangle, Circle
from TrajectryGeneration.objects.Point2D import Point2D
from TrajectryGeneration.RRT import RRT_star, RRT

sys.path.append(os.path.join(os.path.dirname(__file__), '/TrajectryGeneration'))


def field_generator(N: int, object_num: int, point_num: int) -> (list[Field], list[list[Point2D]]):
    # N個のfieldをランダムに作成
    L: float = 20.0
    fields: list[None | Field] = [None] * N
    points: list[list[None | Point2D]] = [[None] * point_num for _ in range(N)]
    obj_points = np.random.rand(N, object_num, 2) * L - L / 2.0

    for i in range(N):
        tmp_field = Field(w=L, h=L, center=True)
        objects = [Rectangle(x=pt[0], y=pt[1], w=5.0, h=5.0, theta=0.0, fill=True) for pt in obj_points[i]]
        tmp_field.obstacles += objects
        index: int = 0
        while index < point_num:
            xy = np.random.rand(2) * L - L / 2.0
            if not tmp_field.check_collision(Circle(xy[0], xy[1], 0.1)):
                points[i][index] = Point2D(xy[0], xy[1])
                index += 1
        fields[i] = tmp_field

    return fields, points


def generate_point_cloud(field: Field, num_pc: int, object_num: int) -> np.array:
    each_pc_size = int(num_pc / object_num)
    point_clouds = np.random.rand(object_num, each_pc_size, 2)
    for i in range(len(field.obstacles)):
        x, y = field.obstacles[i].pos.x, field.obstacles[i].pos.y
        w, h = field.obstacles[i].w, field.obstacles[i].h
        point_clouds[i, :, 0] *= w
        point_clouds[i, :, 0] += x - w / 2.0
        point_clouds[i, :, 1] *= h
        point_clouds[i, :, 1] += y - h / 2.0
    point_clouds = np.concatenate(point_clouds, axis=0)
    return point_clouds


def gen_test_data(fields, points):
    ans_paths = []
    for i in tqdm(range(len(fields))):
        rrt_star = RRT_star(fields[i], None, R=20.0, eps=0.25, goal_sample_rate=0.05, check_length=None)  # RRT*
        paths = []
        perm = itertools.permutations(points[i], 2)
        for start_goal in perm:
            if fields[i].check_collision_line_segment(start_goal[0], start_goal[1]):
                _, path, _ = rrt_star.planning(start_goal[0], start_goal[1], 150, show=False)
                if path is not None:
                    paths.append(path)
            else:
                paths.append(start_goal)
        ans_paths.append(paths)
    return ans_paths


if __name__ == '__main__':
    fields, points = field_generator(2, 7, 5)
    datas = gen_test_data(fields, points)

    for i in range(len(fields)):
        path = datas[i][0]
        ax = fields[i].plot_path_control_point(path, show=False)
        pcs = generate_point_cloud(fields[i], 2800, 7)
        ax.scatter(pcs[:, 0], pcs[:, 1], color="blue", s=1.0)
        for node in points[i]:
            ax.plot(node.x, node.y, color="red", marker='x', markersize=5.0)
        plt.show()
