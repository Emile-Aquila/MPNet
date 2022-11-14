import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from TrajectryGeneration.models import Robot_model
from TrajectryGeneration.objects.field import Field, Rectangle
from TrajectryGeneration.objects.Point2D import Point2D
# from TrajectryGeneration.RRT import RRT_star

sys.path.append(os.path.join(os.path.dirname(__file__), '/TrajectryGeneration'))


def field_generator(N: int, object_num: int) -> list[Field]:
    # N個のfieldをランダムに作成
    fields: list[None | Field] = [None] * N
    obj_points = np.random.rand(N, object_num, 2) * 20.0 - 10.0

    for i in range(N):
        tmp_field = Field(w=100.0, h=100.0)
        objects = [Rectangle(x=pt[0], y=pt[1], w=5.0, h=5.0, theta=0.0) for pt in obj_points[i]]
        tmp_field.obstacles += objects
        fields[i] = tmp_field

    return fields


if __name__ == '__main__':
    test = field_generator(2, 2)
    test[0].plot()
