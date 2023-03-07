from manimlib import *
import numpy as np
import torch
from playground_v2 import ManimNeuralNet

class MyButton(ControlMobject):
    def __init__(self, value: int = 0, *args, **kwargs):
        self.number = Integer(value)
        super().__init__(value, self.number, *args, **kwargs)
        self.add_mouse_press_listner(self.on_mouse_press)

    def assert_value(self, value):
        assert(isinstance(value, np.float64))

    def set_value_anim(self, value):
        self.number.set_value(self.number.get_value() + 1)

    def toggle_value(self) -> None:
        super().set_value(self.get_value() + 1)

    def on_mouse_press(self, mob: Mobject, event_data) -> bool:
        print('a')
        mob.toggle_value()
        return False



class TestControlM(Scene):
    def construct(self) -> None:
        test = SpiralDataSet()
        self.add(test)


class HiddenLayerControl(ControlMobject):
    def __init__(self, value: int = 0, *args, **kwargs):
        self.number = Integer(value)
        self.Text = TexText("Hidden Layers")
        self.NumberText = MTexText(f"{self.number.get_value()}").add_updater(
            lambda mob: mob.become(MTexText(f"{self.number.get_value()}")))
        always(self.Text.next_to, self.NumberText, RIGHT)
        super().__init__(value, self.Text, self.NumberText, *args, **kwargs)
        self.add_mouse_press_listner(self.on_mouse_press)

    def assert_value(self, value):
        pass

    def set_value_anim(self, value):
        self.number.set_value(value)

    def toggle_value(self):
        super().set_value(self.number.get_value() + 1)

    def on_mouse_press(self, mob: Mobject, event_data):
        mob.toggle_value()
        return False


class DataSet(VGroup):
    get_pressed = 0
    def __init__(self, n=100, c=2, d=2, std=0.2, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.num_points = n
        self.num_class = c
        self.dimension = d
        self.std = std
        self.colors = [RED, BLUE, PURPLE, YELLOW, PINK]
        self.DataPoints = VGroup()

        self.X = torch.zeros(self.num_points * self.num_class, self.dimension)
        self.y = torch.zeros(self.num_points * self.num_class, dtype=torch.long)

        self.add_datapoints()
        self.add_bounding_rec()
        self.add_mouse_press_listner(self.on_mouse_press)

    @abstractmethod
    def generate_data(self):
        raise Exception(self.__getattribute__("generate_data") + "Not Defined")

    @abstractmethod
    def add_datapoints(self):
        self.add(self.DataPoints)

    def add_bounding_rec(self):
        self.BoudingRec = SurroundingRectangle(self.DataPoints, color=WHITE, stroke_width=1).scale(1.2)
        self.add(self.BoudingRec)

    def add_datapoints(self):
        self.generate_data()
        X = self.X.numpy()
        y = self.y.numpy()

        self.DataPoints.add(
            *[Dot(point=np.array([X[i, 0], X[i, 1], 0]), color=self.colors[y[i]], radius=0.01)
              for i in range(X.shape[0])]
        )
        self.add(self.DataPoints)

    def on_mouse_press(self, mob: Mobject):
        mob.get_pressed += 1
        return False


class SpiralDataSet(DataSet):
    def generate_data(self):
        for i in range(self.num_class):
            index = 0
            r = torch.linspace(0.2, 1, self.num_points)
            t = torch.linspace(
                i * 2 * math.pi / self.num_class,
                (i + 2) * 2 * math.pi / self.num_class,
                self.num_points
            ) + torch.randn(self.num_points) * self.std

            for ix in range(self.num_points * i, self.num_points * (i + 1)):
                self.X[ix] = r[index] * torch.FloatTensor((
                    math.sin(t[index]), math.cos(t[index])
                ))
                self.y[ix] = i
                index += 1





