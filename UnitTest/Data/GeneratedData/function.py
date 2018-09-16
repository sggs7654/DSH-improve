import Data.GeneratedData
import General.draw


def build_center_test():
    gd = Data.GeneratedData.GeneratedData()
    gd.build_center()
    draw = General.draw.draw()
    draw.point_set(gd.center_set)


def build_point_test():
    gd = Data.GeneratedData.GeneratedData()
    gd.build_point()
    draw = General.draw.draw()
    draw.point_set(gd.point_set)
