import torch
from manimlib import *


class Neuron(VGroup):
    neuron_radius = 0.3
    neuron_stroke_width = 1

    def __init__(self, label=None, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.Circle = Circle(
            radius=self.neuron_radius,
            stroke_width=self.neuron_stroke_width,
            color=BLUE_B,
            opacity=0.7
        ).set_fill(BLUE, opacity=0.7)
        self.add(self.Circle)
        # self.weight = np.random.normal(size=(2, 1))
        # self.dots = VGroup()
        # self.points = self.get_points()
        self.edge_in = VGroup()
        self.edge_out = VGroup()
        if label is not None:
            self.Label = MTexText(label).scale(2)
            always(self.Label.next_to, self.Circle, LEFT)
            self.add(self.Label)

        # self.add_updater(lambda mob: mob.update_weights())
        # self.add_updater(lambda mob: mob.update_points())
        # self.add_to_back(self.dots)

    # def update_points(self):
    #     for i in range(len(self.dots)):
    #         if self.points[i][:2].dot(self.weight) > 0:
    #             self.dots[i].set_color(RED)
    #         elif self.points[i][:2].dot(self.weight) == 0:
    #             self.dots[i].set_color(WHITE)
    #         else:
    #             self.dots[i].set_color(BLUE)
    #
    # def get_points(self):
    #     self.dots = VGroup()
    #     x = np.linspace(-1, 1, 10)
    #     y = np.linspace(-1, 1, 10)
    #     xx, yy = np.meshgrid(x, y)
    #     points = np.c_[np.ravel(xx), np.ravel(yy), np.zeros(np.ravel(xx).shape)]
    #     for point in points:
    #         if point[:2].dot(self.weight) > 0:
    #             dot = Dot(point, radius=0.2).set_color(RED)
    #         elif point[:2].dot(self.weight) == 0:
    #             dot = Dot(point, radius=0.2).set_color(WHITE)
    #         else:
    #             dot = Dot(point, radius=0.2).set_color(BLUE)
    #         self.dots.add(dot)
    #     return points
    #
    # def update_weights(self):
    #     self.weight = np.random.normal(size=(2, 1))


class TestScene(Scene):
    def construct(self) -> None:
        n = Neuron('$x_{1}$')
        n.to_edge(LEFT)
        n.to_edge(RIGHT)
        n.scale(0.5)
        self.add(n)


class MyButton(VGroup):
    def __init__(self, content: str, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.ContentText = Text(content)
        self.ContentCircle = Circle(color=WHITE, stroke_width=3)
        self.ContentCircle.surround(self.ContentText, buff=0.2)
        self.add(self.ContentText)
        self.add(self.ContentCircle)

    # def get_bounding_box(self) -> np.ndarray:
    #     return self.ContentText.get_bounding_box()


class HiddenLayerControl(ControlMobject):
    def __init__(self, value=0, text='neurons', max_num=5, *args, **kwargs):
        self.PlusButton = MyButton('+').scale(0.6)
        self.MinusButton = MyButton('-').scale(0.8)
        self.Buttons = VGroup(*[self.PlusButton, self.MinusButton]).scale(0.6)
        self.Buttons.arrange(RIGHT)
        self.number = Integer(value)
        self.Text = TexText(text, color=WHITE).scale(0.6)
        self.get_pressed = True
        self.max_num = max_num
        self.NumberText = MTexText(f"{self.number.get_value()}").scale(0.7).add_updater(
            lambda mob: mob.become(MTexText(f"{self.number.get_value()}").scale(0.7)))
        always(self.Text.next_to, self.NumberText, RIGHT)
        always(self.NumberText.next_to, self.Buttons, DOWN + LEFT)
        super().__init__(value, self.NumberText, self.Text, self.Buttons, *args, **kwargs)
        self.add_mouse_press_listner(self.on_mouse_press)

    def set_value_anim(self, value):
        self.number.set_value(value)

    def toggle_value(self, point):
        plus_bounding = self.PlusButton.get_bounding_box()
        minus_bounding = self.MinusButton.get_bounding_box()
        print(point, plus_bounding, minus_bounding)
        if plus_bounding[0][0] < point[0] < plus_bounding[2][0] and \
                plus_bounding[0][1] < point[1] < plus_bounding[2][1] \
                and self.number.get_value() < self.max_num:
            super().set_value(self.number.get_value() + 1)
            self.get_pressed = True
        elif minus_bounding[0][0] < point[0] < minus_bounding[2][0] and \
                minus_bounding[0][1] < point[1] < minus_bounding[2][1] \
                and self.number.get_value() > 0:
            super().set_value(self.number.get_value() - 1)
            self.get_pressed = True
        else:
            self.get_pressed = False

    def on_mouse_press(self, mob: Mobject, event_data):
        mob.toggle_value(event_data['point'])
        return False


class NeuralLayer(VGroup):
    def __init__(self, num_neurons, labels=None, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        layer = self.get_layer(num_neurons)
        self.layer = layer
        self.num_neurons = num_neurons
        self.add(self.layer)

    @staticmethod
    def get_layer(num_neurons):
        if num_neurons > 0:
            layer = VGroup()
            for i in range(num_neurons):
                layer.add(Neuron())
            layer.arrange(DOWN, buff=MED_SMALL_BUFF)
            return layer
        return VGroup()


class NeuralNet(VGroup):
    def __init__(self, layers_size, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.layers_size = layers_size
        self.add_layers()

    def add_layers(self):
        layers = VGroup()
        for layer_size in self.layers_size:
            layer = NeuralLayer(layer_size)
            layers.add(layer)
        layers.arrange(RIGHT, buff=LARGE_BUFF * 2)
        self.add(layers)
        self.layers = layers


class ControlPanel(Scene):
    def construct(self) -> None:
        test = NeuralLayer(2)
        self.add(test)


class TestNeuralNet(Scene):
    def construct(self) -> None:
        # control_panel = HiddenLayerControl(1, 'hidden layers').move_to(np.array([0, 2.5, 0]))
        # brace = always_redraw(Brace, control_panel, UP)
        # self.add(control_panel, brace)
        first_button = HiddenLayerControl(2).to_edge(UL)
        last_button = HiddenLayerControl(2).to_edge(UR)
        self.add(first_button)
        self.add(last_button)

        input_layer = NeuralLayer(2).next_to(first_button, DOWN)
        output_layer = NeuralLayer(2).next_to(last_button, DOWN)
        self.add(input_layer, output_layer)

        input_layer.add_updater(lambda mob: mob.become(NeuralLayer(first_button.number.get_value()))
                                .next_to(first_button, DOWN))
        output_layer.add_updater(lambda mob: mob.become(NeuralLayer(last_button.number.get_value()))
                                 .next_to(last_button, DOWN))

        hidden_layers_size = [2, 4, 2]

        distance = (last_button.get_center() - first_button.get_center()) / 4
        first_hidden_button = HiddenLayerControl(hidden_layers_size[0]).move_to(distance + first_button.get_center())
        second_hidden_button = HiddenLayerControl(hidden_layers_size[1]).move_to(
            distance * 2 + first_button.get_center())
        third_hidden_button = HiddenLayerControl(hidden_layers_size[2]).move_to(
            distance * 3 + first_button.get_center())
        self.add(first_hidden_button, second_hidden_button, third_hidden_button)

        first_hidden_layer = NeuralLayer(hidden_layers_size[0]).add_updater(
            lambda mob: mob.become(NeuralLayer(
                first_hidden_button.number.get_value())).next_to(first_hidden_button, DOWN)
        )
        second_hidden_layer = NeuralLayer(hidden_layers_size[1]).add_updater(
            lambda mob: mob.become(NeuralLayer(
                second_hidden_button.number.get_value())).next_to(second_hidden_button, DOWN)
        )
        third_hidden_layer = NeuralLayer(hidden_layers_size[2]).add_updater(
            lambda mob: mob.become(NeuralLayer(
                third_hidden_button.number.get_value())).next_to(third_hidden_button, DOWN)
        )
        self.add(first_hidden_layer, second_hidden_layer, third_hidden_layer)


class TestControl(Scene):
    def construct(self) -> None:
        self.panel = HiddenLayerControl(0, text='hidden layers', max_num=5).move_to(ORIGIN + UP * 2.5)
        self.add(self.panel)
        self.first_button = HiddenLayerControl(2).move_to(ORIGIN + UP * 1.5 + LEFT * 5)
        self.last_button = HiddenLayerControl(2).move_to(ORIGIN + UP * 1.5 + RIGHT * 5)
        self.add(self.first_button)
        self.add(self.last_button)

        input_layer = NeuralLayer(2).next_to(self.first_button, DOWN)
        output_layer = NeuralLayer(2).next_to(self.last_button, DOWN)
        self.add(input_layer, output_layer)
        self.remove(self.last_button)

        input_layer.add_updater(lambda mob: mob.become(NeuralLayer(self.first_button.number.get_value()))
                                .next_to(self.first_button, DOWN))
        # output_layer.add_updater(lambda mob: mob.become(NeuralLayer(last_button.number.get_value()))
        #                          .next_to(last_button, DOWN))
        self.hidden_layers_size = []
        self.hidden_layers = VGroup()
        self.add(self.hidden_layers)
        self.hidden_layers_num = 0

    def hidden_layer_updater(self, hidden_button, hidden_num):
        hidden_button.move_to(self.first_button.get_center() + hidden_num * self.pos_buff)
        if hidden_button.number.get_value() == 0:
            self.remove(hidden_button)
            hidden_button.become(VGroup())
            self.hidden_layers_num -= 1
            self.construct_hidden_layers(self.hidden_layers_num)
            self.panel.number.set_value(self.hidden_layers_num)

    def construct_hidden_layers(self, hidden_num):
        # self.remove(self.hidden_layers)
        # self.hidden_layers.remove(*self.hidden_layers)
        self.pos_buff = (self.last_button.get_center() - self.first_button.get_center()) / (hidden_num + 1)
        if hidden_num > self.hidden_layers_num:
            if hidden_num == 1:
                self.first_hidden_button = HiddenLayerControl(2).add_updater(
                    lambda mob: self.hidden_layer_updater(mob, 1)
                )

                self.first_hidden_layer = NeuralLayer(2).add_updater(
                    lambda mob: mob.become(NeuralLayer(
                        self.first_hidden_button.number.get_value())).next_to(self.first_hidden_button, DOWN)
                )
                self.add(self.first_hidden_button, self.first_hidden_layer)
                self.hidden_layers_num = 1
            elif hidden_num == 2:
                # self.remove(self.first_hidden_button)
                # self.first_hidden_button.move_to(self.first_button.get_center() + self.pos_buff)
                # self.add(self.first_hidden_button)
                self.second_hidden_button = HiddenLayerControl(2).add_updater(
                    lambda mob: self.hidden_layer_updater(mob, 2)
                )
                self.second_hidden_layer = NeuralLayer(2).add_updater(
                    lambda mob: mob.become(NeuralLayer(
                        self.second_hidden_button.number.get_value())).next_to(self.second_hidden_button, DOWN)
                )
                self.add(self.second_hidden_button, self.second_hidden_layer)
                self.hidden_layers_num = 2

    def on_mouse_press(
            self,
            point: np.ndarray,
            button: int,
            mods: int
    ) -> None:
        prev_panel_num = self.panel.number.get_value()
        super().on_mouse_press(point, button, mods)
        now_panel_num = self.panel.number.get_value()
        if prev_panel_num != now_panel_num:
            print('change')
            self.construct_hidden_layers(now_panel_num)


class HiddenLayerControlSet(ControlMobject):
    def __init__(self, value: int = 2, max_num=5, text='neurons', *args, **kwargs):
        self.PlusButton = MyButton('+').scale(0.6)
        self.MinusButton = MyButton('-').scale(0.8)
        self.Buttons = VGroup(*[self.PlusButton, self.MinusButton]).scale(0.6)
        self.Buttons.arrange(RIGHT)
        self.number = Integer(value)
        self.Text = TexText(text, color=WHITE).scale(0.6)
        self.get_pressed = True
        self.max_num = max_num
        self.NumberText = MTexText(f"{self.number.get_value()}").scale(0.7).add_updater(
            lambda mob: mob.become(MTexText(f"{self.number.get_value()}").scale(0.7)))
        always(self.Text.next_to, self.NumberText, RIGHT)
        always(self.NumberText.next_to, self.Buttons, DOWN + LEFT)

        self.layer = NeuralLayer(value)
        always(self.layer.next_to, self.Text, DOWN)

        super().__init__(value, self.NumberText, self.Text, self.Buttons, self.layer, *args, **kwargs)
        self.add_mouse_press_listner(self.on_mouse_press)

    def set_value_anim(self, value):
        self.number.set_value(value)
        self.layer.become(NeuralLayer(value))

    def toggle_value(self, point):
        plus_bounding = self.PlusButton.get_bounding_box()
        minus_bounding = self.MinusButton.get_bounding_box()
        print(point, plus_bounding, minus_bounding)
        if plus_bounding[0][0] < point[0] < plus_bounding[2][0] and \
                plus_bounding[0][1] < point[1] < plus_bounding[2][1] \
                and self.number.get_value() < self.max_num:
            super().set_value(self.number.get_value() + 1)
            self.get_pressed = True
        elif minus_bounding[0][0] < point[0] < minus_bounding[2][0] and \
                minus_bounding[0][1] < point[1] < minus_bounding[2][1] \
                and self.number.get_value() > 0:
            super().set_value(self.number.get_value() - 1)
            self.get_pressed = True
        else:
            self.get_pressed = False

    def on_mouse_press(self, mob: Mobject, event_data):
        mob.toggle_value(event_data['point'])
        return False


class HiddenLayerControlPanel(ControlMobject):
    def __init__(self, value: int = 2, max_num=5, text='neurons', *args, **kwargs):
        self.PlusButton = MyButton('+').scale(0.6)
        self.MinusButton = MyButton('-').scale(0.8)
        self.Buttons = VGroup(*[self.PlusButton, self.MinusButton]).scale(0.6)
        self.Buttons.arrange(RIGHT)
        self.number = Integer(value)
        self.Text = TexText(text, color=WHITE).scale(0.6)
        self.get_pressed = True
        self.max_num = max_num
        self.NumberText = MTexText(f"{self.number.get_value()}").scale(0.7).add_updater(
            lambda mob: mob.become(MTexText(f"{self.number.get_value()}").scale(0.7)))
        always(self.Text.next_to, self.NumberText, RIGHT)
        always(self.NumberText.next_to, self.Buttons, DOWN + LEFT)



class TestControlSet(Scene):
    def construct(self) -> None:
        test = HiddenLayerControlSet(2)
        self.add(test)
