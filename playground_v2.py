import manimlib
from manimlib import *
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32


class DataSet(VGroup):
    get_pressed = 0
    check_pressed = False
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

    def on_mouse_press(self, mob:Mobject, event_data):
        if mob.check_pressed:
            mob.get_pressed += 1
            print('pressed')
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


class ClusterDataSet(DataSet):
    def generate_data(self):
        sign = 1
        for i in range(self.num_class):
            for ix in range(self.num_points * i, self.num_points * (i + 1)):
                self.X[ix] = torch.FloatTensor(torch.normal(mean=0.5 * sign, std=self.std, size=(1, 2)))
                self.y[ix] = i
            sign *= -1


class TestDataSet(Scene):
    def construct(self):
        data = SpiralDataSet()
        self.play(Write(data))
        self.wait()


class NeuralNet(nn.Sequential):
    def __init__(self, layers_size, dropout=1, batchnorm=True, activation='ReLU'):
        super().__init__()
        self.layers_size = layers_size
        for i in range(len(layers_size) - 1):
            layer = nn.Sequential()
            if batchnorm:
                layer.add_module(module=nn.BatchNorm1d(layers_size[i]), name=f"BatchNorm1d{i}")
            layer.add_module(module=nn.Linear(layers_size[i], layers_size[i + 1]), name=f"Linear{i}")
            if activation == 'ReLU':
                layer.add_module(module=nn.ReLU(inplace=True), name=f"{activation}{i}")
            elif activation == 'Sigmoid':
                layer.add_module(module=nn.Sigmoid(), name=f"{activation}{i}")
            else:
                layer.add_module(module=nn.Tanh(), name=f"{activation}{i}")
            if dropout < 1:
                layer.add_module(module=nn.Dropout(p=dropout, inplace=True), name=f"Dropout{i}")

            self.add_module(module=layer, name=f"LinearBlock{i}")


class ActivationFunction(VGroup):
    get_pressed = 0
    check_pressed = False
    def __init__(self,
                 function_name=None,
                 x_range=[-1, 1],
                 y_range=[-1, 1],
                 x_length=0.5,
                 y_length=0.3,
                 show_function_name=True,
                 activate_color=BLUE,
                 bounding_box_color=WHITE):
        super(VGroup, self).__init__()
        self.x_range = x_range
        self.y_range = y_range
        self.x_length = x_length
        self.y_length = y_length

        self.function_name = function_name
        self.show_function_name = show_function_name
        self.activate_color = activate_color
        self.bounding_box_color = bounding_box_color

        self.add_activation_function()
        self.add_mouse_press_listner(self.on_mouse_press)
        self.scale(0.25)

    def add_activation_function(self):
        axes = Axes(
            x_range=self.x_range,
            y_range=self.y_range,
            x_length=self.x_length,
            y_length=self.y_length,
            tips=False,
            axis_config={
                "include_numbers": False,
                "stroke_width": 2,
                "include_ticks": False,
                "color": WHITE
            }
        )

        activation_graph = axes.get_graph(
            lambda x: self.apply_activate_function(x),
            using_smooth=False,
            color=self.activate_color
        )

        # if self.show_function_name and self.function_name is not None:
        #     activation_label = axes.get_graph_label(activation_graph, Text(self.function_name))
        #     always(activation_label.next_to, activation_graph, DOWN)
        # else:
        #     activation_label = None
        activation_label = None
        self.activation_and_axes = VGroup(*[activation_graph, axes])

        BoundingRec = SurroundingRectangle(self.activation_and_axes, color=self.bounding_box_color,
                                           stroke_width=1).scale(1.2)

        self.activation_mobject = VGroup(*[self.activation_and_axes, BoundingRec])
        self.add(self.activation_mobject)

    def apply_activate_function(self, x):
        raise Exception("Activation Function Not Defined")

    def on_mouse_press(self, mob:Mobject, event_data):
        if mob.check_pressed:
            mob.get_pressed += 1
        print('pressed')
        return False


class ReLUActivation(ActivationFunction):
    def __init__(self):
        super().__init__(function_name='ReLU')

    def apply_activate_function(self, x):
        return x if x > 0 else 0


class SigmoidActivation(ActivationFunction):
    def __init__(self):
        super().__init__(function_name='Sigmoid',
                         x_range=[-10, 10],
                         y_range=[0, 1])

    def apply_activate_function(self, x):
        return 1 / (1 + math.exp(-x))


class TanhActivation(ActivationFunction):
    def __init__(self):
        super().__init__(function_name='Tanh',
                         x_range=[-10, 10])

    def apply_activate_function(self, x):
        return (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)


class ModuleBlock(VGroup):
    get_pressed = 0
    check_pressed = False

    def __init__(self, batchnorm=True, dropout=True, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.batchnorm = batchnorm
        self.dropout = dropout
        if self.batchnorm:
            self.BatchNormText = TexText(
                """
                BatchNorm 1d\n
                """
            )
        else:
            self.BatchNormText = TexText(
                """
                BatchNorm 1d\n
                """,
                color="#333333"
            )
        self.LinearText = TexText(
            """
            Activation(Linear)
            """
        )
        if self.dropout:
            self.DropoutText = TexText(
                """
                Dropout\n
                """
            )
        else:
            self.DropoutText = TexText(
                """
                Dropout\n
                """,
                color="#333333"
            )
        self.add(*[self.BatchNormText, self.LinearText, self.DropoutText])
        self.arrange(DOWN)
        print(VGroup(*[self.BatchNormText, self.LinearText, self.DropoutText]).get_bounding_box())
        self.add(SurroundingRectangle(VGroup(*[self.BatchNormText, self.LinearText, self.DropoutText]),
                                      color=WHITE, stroke_width=1).scale(1.2))
        self.add_mouse_press_listner(self.on_mouse_press)

    def on_mouse_press(self, mob:Mobject, event_data):
        if mob.check_pressed:
            mob.get_pressed += 1
        print('here')
        return False


class TestModule(Scene):
    def construct(self) -> None:
        test = ModuleBlock(False, False)
        self.play(ShowCreation(test))


class ManimNeuralNet(VGroup):
    neuron_radius = 0.3
    neuron_to_neuron_buff = MED_SMALL_BUFF
    neuron_stroke_width = 1
    neuron_stroke_color = WHITE
    layer_to_layer_buff = LARGE_BUFF
    edge_color = GREY_C
    edge_stroke_width = 2

    def __init__(self, layers_size, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.hidden_layers = layers_size
        self.add_neurons()
        self.add_edges()
        self.add_to_back(self.layers)

    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size)
            for size in self.hidden_layers
        ])
        layers.arrange(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers

    def get_layer(self, size):
        layer = VGroup()
        neurons = VGroup(*[
            Circle(
                radius=self.neuron_radius,
                stroke_width=self.neuron_stroke_width,
                stroke_color=self.neuron_stroke_color,
                fill_color=WHITE
            )
            for _ in range(size)
        ])
        neurons.arrange(DOWN, buff=self.neuron_to_neuron_buff)

        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        return layer

    def add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)

    def get_edge(self, neuron1: Circle, neuron2: Circle):
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff=self.neuron_radius,
            stroke_color=self.edge_color,
            stroke_width=self.edge_stroke_width,
        )


class MyButton(VGroup):
    def __init__(self, content: str, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.ContentText = Text(content)
        self.ContentCircle = Circle(color=WHITE, stroke_width=3)
        self.ContentCircle.surround(self.ContentText, buff=0.2)
        self.add(self.ContentText)
        self.add(self.ContentCircle)

    def get_bounding_box(self) -> np.ndarray:
        return self.ContentText.get_bounding_box()


class TestButton(Scene):
    def construct(self) -> None:
        test = EnableDisableButton()
        self.add(test)


class HiddenLayerControl(ControlMobject):
    def __init__(self, value=0, *args, **kwargs):
        self.PlusButton = MyButton('+').scale(0.8)
        self.MinusButton = MyButton('-')
        self.Buttons = VGroup(*[self.PlusButton, self.MinusButton])
        self.Buttons.arrange(RIGHT * 3)
        self.number = Integer(value)
        self.Text = TexText("Hidden Layers")
        self.get_pressed = True
        self.NumberText = MTexText(f"{self.number.get_value()}").add_updater(
            lambda mob: mob.become(MTexText(f"{self.number.get_value()}")))
        always(self.Text.next_to, self.NumberText, RIGHT)
        always(self.NumberText.next_to, self.Buttons, RIGHT)
        super().__init__(value, self.NumberText, self.Text, self.Buttons, *args, **kwargs)
        self.add_mouse_press_listner(self.on_mouse_press)

    def set_value_anim(self, value):
        self.number.set_value(value)

    def toggle_value(self, point):
        plus_bounding = self.PlusButton.get_bounding_box()
        minus_bounding = self.MinusButton.get_bounding_box()
        print(point, plus_bounding, minus_bounding)
        if plus_bounding[0][0] < point[0] < plus_bounding[2][0] and \
                plus_bounding[0][1] < point[1] < plus_bounding[2][1]\
                and self.number.get_value() < 3:
            super().set_value(self.number.get_value() + 1)
            self.get_pressed = True
        elif minus_bounding[0][0] < point[0] < minus_bounding[2][0] and \
                minus_bounding[0][1] < point[1] < minus_bounding[2][1]\
                and self.number.get_value() > 0:
            super().set_value(self.number.get_value() - 1)
            self.get_pressed = True
        else:
            self.get_pressed = False

    def on_mouse_press(self, mob: Mobject, event_data):
        mob.toggle_value(event_data['point'])
        return False


class TestHiddenScene(Scene):
    def construct(self) -> None:
        test = HiddenLayerControl()
        self.add(test)


class PlayGround(Scene):
    intro_ready = False
    press_space_tribute = False
    press_space_intro = False
    press_number_for_data = False
    create_neural_net = False
    press_number_for_active = False
    neural_net_layers = 0
    layers_size = [2]
    neural_net = None
    pick_dataset = False
    pick_activation = False
    pick_module = False
    pick_all = False
    start_pick = False
    start_construct_nn = False

    def construct(self):
        self.play_tribute()

    def play_tribute(self):
        self.TributeText = Text(
            """
            This manimation project is powered by ManimGL,\n
            which was created by Grant Sanderson AKA 3Blue1Brown.\n
            (Press Space To Continue...)
            """,
            font_size=25,
            t2s={"Grant Sanderson": ITALIC},
            t2w={"ManimGL": BOLD},
            t2c={"Blue": BLUE, "Grant Sanderson": BLUE, "Brown": YELLOW_D}
        )
        self.add(self.TributeText)
        self.play(Write(self.TributeText))

    def play_intro(self):
        self.PlayGroundText = TexText(
            """
            This is Manim PlayGround
            """
        )

        self.PoweredBy = TexText(
            """
            Powered By\n
            ManimGL\n
            (Press Space To Continue...)
            """
        )
        always(self.PoweredBy.next_to, self.PlayGroundText, DOWN)

        self.IntroTexts = VGroup(*[self.PlayGroundText, self.PoweredBy])
        self.IntroTexts.center()
        self.add(self.IntroTexts)
        self.play(Write(self.IntroTexts, run_time=3))

    def add_datasets(self):
        self.DataSetText = TexText(
            """
            DATASET\n
            """
        ).scale(0.7)
        self.DataSetText.to_edge(UL)
        self.play(Write(self.DataSetText))

        self.SpiralData = SpiralDataSet().scale(0.5)
        self.ClusterData = ClusterDataSet().scale(0.5)
        self.DataSets = VGroup(*[self.SpiralData, self.ClusterData])
        self.DataSets.arrange(RIGHT)
        self.DataSets.next_to(self.DataSetText, DOWN)
        self.DataSets.to_edge(LEFT)
        self.play(Write(self.DataSets))

    def add_activation_functions(self):
        self.ActivationText = TexText(
            """
            ACTIVATION FUNCTION\n
            """
        ).scale(0.7)
        self.ActivationText.to_edge(UR)
        self.play(Write(self.ActivationText))

        self.ReLU = ReLUActivation().scale(0.7)
        self.Sigmoid = SigmoidActivation().scale(0.7)
        self.Tanh = TanhActivation().scale(0.7)
        self.ActivationFunctions = VGroup(*[self.ReLU, self.Sigmoid, self.Tanh])
        self.ActivationFunctions.arrange(RIGHT)
        self.ActivationFunctions.next_to(self.ActivationText, DOWN)
        self.ActivationFunctions.to_edge(RIGHT)
        self.play(Write(self.ActivationFunctions))

    def add_module_block(self):
        self.BatchLinearDrop = ModuleBlock()
        self.BatchLinear = ModuleBlock(dropout=False)
        self.LinearDrop = ModuleBlock(batchnorm=False)
        self.Linear = ModuleBlock(False, False)
        self.ModuleText = TexText(
            """
            MODULE BLOCK\n
            """
        ).scale(0.7)
        self.ModuleText.next_to(VGroup(*[self.DataSets, self.ActivationFunctions]), DOWN, MED_LARGE_BUFF)
        self.play(Write(self.ModuleText))

        self.ModuleBlocks = VGroup(
            *[
                self.BatchLinearDrop,
                self.BatchLinear,
                self.LinearDrop,
                self.Linear
            ]
        )
        self.ModuleBlocks.scale(0.6)
        self.ModuleBlocks.arrange(RIGHT)
        self.ModuleBlocks.next_to(self.ModuleText, DOWN)
        self.play(ShowCreation(self.ModuleBlocks))

    def play_pick(self):
        pass

    def check_if_mouse_in(self, mobject: manimlib.VGroup):
        ld, _, ru = mobject.get_bounding_box()
        return ld[0] < self.mouse_point.get_points()[0][0] < ru[0] and \
            ld[1] < self.mouse_point.get_points()[0][1] < ru[1]

    def check_pick_datasets(self):
        if self.pick_dataset is False:
            self.SpiralData.check_pressed = True
            self.ClusterData.check_pressed = True
            if self.check_if_mouse_in(self.SpiralData):
                self.pick_dataset = True
                self.play(FadeOut(self.ClusterData), FadeOut(self.DataSetText))
                self.DataSets.remove(self.ClusterData)
                self.train_data = (self.SpiralData.X, self.SpiralData.y)
                self.DataSet = self.SpiralData
            elif self.check_if_mouse_in(self.ClusterData):
                self.pick_dataset = True
                self.play(FadeOut(self.SpiralData), FadeOut(self.DataSetText))
                self.DataSets.remove(self.SpiralData)
                # self.play(self.ClusterData.animate.to_edge(LEFT))
                self.train_data = (self.ClusterData.X, self.ClusterData.y)
                self.DataSet = self.ClusterData

    def check_pick_activation(self):
        if self.pick_activation is False:
            self.ReLU.check_pressed = self.Sigmoid.check_pressed = self.Tanh.check_pressed = True
            if self.check_if_mouse_in(self.ReLU):
                self.pick_activation = True
                self.play(FadeOut(self.Sigmoid), FadeOut(self.Tanh), FadeOut(self.ActivationText))
                self.ActivationFunctions.remove(self.Sigmoid, self.Tanh)
                # self.play(self.ReLU.animate.to_edge(RIGHT))
                self.activation_name = 'ReLU'
                self.ActivationFunction = self.ReLU
            elif self.check_if_mouse_in(self.Sigmoid):
                self.pick_activation = True
                self.play(FadeOut(self.ReLU), FadeOut(self.Tanh), FadeOut(self.ActivationText))
                self.ActivationFunctions.remove(self.ReLU, self.Tanh)
                # self.play(self.Sigmoid.animate.to_edge(RIGHT))
                self.activation_name = 'Sigmoid'
                self.ActivationFunction = self.Sigmoid
            elif self.check_if_mouse_in(self.Tanh):
                self.pick_activation = True
                self.play(FadeOut(self.Sigmoid), FadeOut(self.ReLU), FadeOut(self.ActivationText))
                self.ActivationFunctions.remove(self.Sigmoid, self.ReLU)
                # self.play(self.Tanh.animate.to_edge(RIGHT))
                self.activation_name = 'Tanh'
                self.ActivationFunction = self.Tanh

    def check_pick_module(self):
        if self.pick_module is False:
            self.BatchLinearDrop.check_pressed = self.BatchLinear.check_pressed = self.Linear.check_pressed \
                                               = self.LinearDrop.check_pressed = True
            if self.check_if_mouse_in(self.BatchLinearDrop):
                self.pick_module = True
                self.play(FadeOut(self.BatchLinear), FadeOut(self.Linear), FadeOut(self.LinearDrop),
                          FadeOut(self.ModuleText))
                self.ModuleBlocks.remove(self.BatchLinear, self.Linear, self.LinearDrop)
                # self.play(self.BatchLinearDrop.animate.to_edge(DOWN))
                self.ModuleBlock = self.BatchLinearDrop
                self.module = {'batchnorm': True, 'dropout': True}
            elif self.check_if_mouse_in(self.BatchLinear):
                self.pick_module = True
                self.play(FadeOut(self.BatchLinearDrop), FadeOut(self.Linear), FadeOut(self.LinearDrop),
                          FadeOut(self.ModuleText))
                self.ModuleBlocks.remove(self.BatchLinearDrop, self.Linear, self.LinearDrop)
                # self.play(self.BatchLinear.animate.to_edge(DOWN))
                self.ModuleBlock = self.BatchLinear
                self.module = {'batchnorm': True, 'dropout': False}
            elif self.check_if_mouse_in(self.Linear):
                self.pick_module = True
                self.play(FadeOut(self.BatchLinearDrop), FadeOut(self.BatchLinear), FadeOut(self.LinearDrop),
                          FadeOut(self.ModuleText))
                self.ModuleBlocks.remove(self.BatchLinearDrop, self.BatchLinear, self.LinearDrop)
                # self.play(self.Linear.animate.to_edge(DOWN))
                self.ModuleBlock = self.Linear
                self.module = {'batchnorm': False, 'dropout': False}
            elif self.check_if_mouse_in(self.LinearDrop):
                self.pick_module = True
                self.play(FadeOut(self.BatchLinearDrop), FadeOut(self.Linear), FadeOut(self.BatchLinear),
                          FadeOut(self.ModuleText))
                self.ModuleBlocks.remove(self.BatchLinearDrop, self.Linear, self.BatchLinear)
                # self.play(self.LinearDrop.animate.to_edge(DOWN))
                self.ModuleBlock = self.LinearDrop
                self.module = {'batchnorm': False, 'dropout': True}

    def check_pick_all(self):
        if (self.pick_dataset and self.pick_activation and self.pick_module) and (self.pick_all is False):
            self.PickedSet = VGroup(*[
                self.DataSet,
                self.ActivationFunction,
                self.ModuleBlock
            ])
            # self.PickedSet.arrange(RIGHT)
            self.play(self.PickedSet.animate.arrange(RIGHT))
            self.play(self.PickedSet.animate.scale(0.7))
            self.play(self.PickedSet.animate.to_edge(UP))
            self.pick_all = True

    def add_plus_minus(self):
        PlusButton = MyButton('+').scale(0.8)
        MinusButton = MyButton('-')
        Buttons = VGroup(*[PlusButton, MinusButton])
        Buttons.arrange(RIGHT)
        return Buttons

    def on_key_press(
            self,
            symbol: int,
            modifiers: int
    ) -> None:
        try:
            char = chr(symbol)
        except OverflowError:
            log.warning("The value of the pressed key is too large.")
            return

        if ord(char) == 32 and not self.press_space_tribute:
            self.play(FadeOut(self.TributeText))
            self.press_space_tribute = True
            self.intro_ready = True
            self.play_intro()

        if ord(char) == 32 and self.intro_ready and not self.press_space_intro:
            self.press_space_intro = True
            self.play(FadeOut(self.IntroTexts))
            self.add_datasets()
            self.add_activation_functions()
            self.add_module_block()
            self.start_pick = True
            return

    def on_mouse_press(
        self,
        point: np.ndarray,
        button: int,
        mods: int
    ) -> None:
        if self.start_pick and self.pick_all is False:
            self.check_pick_datasets()
            self.check_pick_activation()
            self.check_pick_module()
            self.check_pick_all()
            if self.pick_all is True:
                self.HiddenLayerButton = HiddenLayerControl()
                self.HiddenLayerButton.next_to(self.PickedSet, DOWN, SMALL_BUFF)
                self.play(ShowCreation(self.HiddenLayerButton))
                self.NeuralNet = None
                self.NeuralNet = ManimNeuralNet([2] + self.HiddenLayerButton.number.get_value() * [2] + [2])
                self.NeuralNet.next_to(self.HiddenLayerButton, DOWN)
                self.play(ShowCreation(self.NeuralNet))

        if self.pick_all is True:
            self.HiddenLayerButton.on_mouse_press(self.HiddenLayerButton, {'point': point})
            if self.HiddenLayerButton.get_pressed:
                self.play(FadeOut(self.NeuralNet))
                self.NeuralNet = ManimNeuralNet([2] + self.HiddenLayerButton.number.get_value() * [2] + [2])
                self.NeuralNet.next_to(self.HiddenLayerButton, DOWN)
                self.play(ShowCreation(self.NeuralNet))


