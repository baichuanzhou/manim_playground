import torch
from manimlib import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32


def get_data_spiral(n=100, d=2, c=2, std=0.2):
    X = torch.zeros(n * c, d)
    y = torch.zeros(n * c, dtype=torch.long)
    for i in range(c):
        index = 0
        r = torch.linspace(0.2, 1, n)
        t = torch.linspace(
            i * 2 * math.pi / c,
            (i + 2) * 2 * math.pi / c,
            n
        ) + torch.randn(n) * std

        for ix in range(n * i, n * (i + 1)):
            X[ix] = r[index] * torch.FloatTensor((
                math.sin(t[index]), math.cos(t[index])
            ))
            y[ix] = i
            index += 1

    return X.numpy(), y.numpy()


def get_data_cluster(n=100, d=2, c=2, std=0.2):
    X = torch.zeros(n * c, d)
    y = torch.zeros(n * c, dtype=torch.long)
    sign = 1
    for i in range(c):
        for ix in range(n * i, n * (i + 1)):
            X[ix] = torch.FloatTensor(torch.normal(mean=0.5 * sign, std=std, size=(1, 2)))
            y[ix] = i
        sign *= -1

    return X.numpy(), y.numpy()


class RealNN(nn.Module):
    def __init__(self, layers_size, activation='relu'):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(layers_size) - 1):
            self.layers.add_module(module=nn.Linear(layers_size[i], layers_size[i + 1]), name=f"Linear{i}")
            self.layers.add_module(module=nn.ReLU(inplace=True), name=f"ReLU{i}")


class DataSet(VGroup):
    def __init__(self, X, y, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        DataPoints = VGroup()
        colors = [RED, BLUE, PURPLE, YELLOW, PINK]
        DataPoints.add(*[Dot(point=np.array([X[i, 0], X[i, 1], 0]), color=colors[y[i]], radius=0.01)
                         for i in range(X.shape[0])])
        self.add(DataPoints)

        ld, mid, ur = self.get_bounding_box()
        height = ur[1] - ld[1] + 0.5
        width = ur[0] - ld[0] + 0.5
        BoundingRec = Rectangle(height=height, width=width, stroke_width=2, color=GREY)
        # always(lambda mob: mob.move_to(DataPoints.get_center()), BoundingRec)
        BoundingRec.add_updater(lambda mob: mob.move_to(DataPoints.get_center()))
        self.add(BoundingRec)


class PlayGround(Scene):
    press_space_intro = False
    press_number_for_data = False
    create_neural_net = False
    press_number_for_active = False
    neural_net_layers = 0
    layers_size = [2]
    neural_net = None

    def construct(self):
        self.wait()
        self.play_intro()

    def play_intro(self):
        self.IntroText = Text(
            """
            This manimation project is powered by ManimGL,\n
            supported by Grant Sanderson AKA 3b1b.\n
            In this project we will visualize the training\n
            process of a basic neural network.\n
            """,
            font_size=25,
            t2s={"Grant Sanderson": ITALIC},
            t2w={"ManimGL": BOLD},
            t2c={"3b1b": BLUE}
        )
        self.add(self.IntroText)
        self.wait(5)
        self.play(FadeOut(self.IntroText))
        self.ThisText = TexText(
            """
            This is Manim PlayGround
            """
        )
        self.play(Write(self.ThisText))

        self.wait()
        self.PoweredBy = TexText(
            """
            Powered By\n
            ManimGL\n
            (Press Space To Continue...)
            """
        )
        always(self.PoweredBy.next_to, self.ThisText, DOWN)
        self.play(Write(self.PoweredBy))
        self.wait()

    def play_datasets(self):
        self.ChooseDataText = TexText(
            """
            Please choose a dataset for training
            """)

        self.play(Write(self.ChooseDataText))

        self.datasets = VGroup()
        self.datatexts = VGroup()

        SpiralData = DataSet(*get_data_spiral())
        SpiralText = TexText("1. Spiral").scale(0.7)
        always(SpiralText.next_to, SpiralData, DOWN)

        ClusterData = DataSet(*get_data_cluster())
        ClusterText = TexText("2. Cluster").scale(0.7)
        always(ClusterText.next_to, ClusterData, DOWN)

        self.datasets.add(SpiralData, ClusterData)
        self.datatexts.add(SpiralText, ClusterText)
        for i, (data, text) in enumerate(zip(self.datasets, self.datatexts)):
            if i == 0:
                move_left = 1
            else:
                move_left = move_left * -1
            if i < len(self.datasets) / 2:
                move_up = 1
            else:
                move_up = -1
            self.play(FadeIn(data), FadeIn(text))
            self.play(data.animate.to_edge(UP * move_up))
            self.play(data.animate.to_edge(LEFT * move_left))
            self.wait()

    def choose_net_layers(self):
        self.ChooseLayerText = TexText(
            """
            Now we create our neural network.\n
            Four layers top, eight neurons max each layer.
            """
        ).scale(0.7)

        self.play(Write(self.ChooseLayerText))
        self.play(self.ChooseLayerText.animate.to_edge(UP, buff=MED_SMALL_BUFF))

    def play_neural_net(self, layers_size, center=False):
        if self.neural_net is not None:
            self.play(FadeOut(self.neural_net))
        self.neural_net = NeuralNet(layers_size)
        if not center:
            self.neural_net.next_to(self.ChooseLayerText, DOWN)
        self.model = RealNN(layers_size)
        self.play(Write(self.neural_net))

    def play_activation_function(self):
        DisplayText = TexText("We provide three types of activation functions")
        DisplayText.next_to(self.dataset, RIGHT, buff=MED_SMALL_BUFF)
        self.play(Write(DisplayText))
        self.play(DisplayText.animate.to_edge(UP))
        ReLU = ReLUActivation()
        Sigmoid = SigmoidActivation()
        Tanh = TanhActivation()
        self.ActivationFunctions = VGroup(*[ReLU, Sigmoid, Tanh])
        self.ActivationFunctions.arrange(RIGHT)
        self.ActivationFunctions.next_to(self.dataset, RIGHT, buff=MED_SMALL_BUFF)
        self.play(ShowCreation(self.ActivationFunctions))
        self.play(FadeOut(DisplayText))

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

        if ord(char) == 32 and not self.press_space_intro:
            self.press_space_intro = True
            self.play(FadeOut(self.ThisText), FadeOut(self.PoweredBy))
            self.play_datasets()
            return

        if ord('0') <= ord(char) <= ord('4') and self.press_space_intro and not self.press_number_for_data:
            self.press_number_for_data = True
            self.play(FadeOut(self.datasets),
                      FadeOut(self.datatexts),
                      FadeOut(self.ChooseDataText))

            self.dataset = VGroup(*[self.datasets[ord(char) - ord('0') - 1], self.datatexts[ord(char) - ord('0') - 1]])
            self.add(self.dataset)

            self.play(Write(self.dataset.center()))
            self.play(self.dataset.animate.to_edge(UL))
            self.play_activation_function()
            return

        # if ord(char) == 32 and self.press_number_for_data and not self.create_neural_net:
        #     self.create_neural_net = True
        #     layers_size = [2, 8, 8, 2]
        #     self.play_neural_net(layers_size)

        if ord('0') <= ord(char) <= ord('3') and self.press_number_for_data and not self.press_number_for_active:
            self.press_number_for_active = True
            self.ActivationFunction = self.ActivationFunctions[ord(char) - ord('0') - 1]
            self.play(FadeOut(self.ActivationFunctions))
            self.play(ShowCreation(self.ActivationFunction.center()))
            self.play(self.ActivationFunction.animate.to_edge(DL, buff=MED_SMALL_BUFF))
            self.choose_net_layers()
            return

        if self.press_number_for_active and not self.create_neural_net and self.neural_net_layers < 4:
            if ord('0') <= ord(char) <= ord('8'):
                NumNeuron = TexText(str(ord(char) - ord('0')))
                NumNeuron.next_to(self.ChooseLayerText, DOWN)
                self.layers_size.append(ord(char) - ord('0'))
                self.neural_net_layers += 1
                self.play_neural_net(self.layers_size)

        if self.neural_net_layers == 4 and not self.create_neural_net:
            self.create_neural_net = True
            self.play(FadeOut(self.ChooseLayerText))
            self.layers_size.append(2)
            self.play_neural_net(self.layers_size, True)
            return




class TestNN(Scene):
    def construct(self):
        TestNeuralNet = NeuralNet([2, 3, 4, 2])
        self.add(TestNeuralNet)
        self.wait()


class NeuralNet(VGroup):
    neuron_radius = 0.3
    neuron_to_neuron_buff = MED_SMALL_BUFF
    neuron_stroke_width = 1
    neuron_stroke_color = WHITE
    layer_to_layer_buff = LARGE_BUFF
    edge_color = GREY_C,
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


class ActivationFunction(VGroup):
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

        self.activation_and_axes = VGroup(*[activation_graph, axes])
        ld, _, ur = self.activation_and_axes.get_bounding_box()
        height = ur[1] - ld[1] + 1
        width = ur[0] - ld[0] + 1
        BoundingRec = Rectangle(height=height, width=width, stroke_width=2, color=self.bounding_box_color)
        BoundingRec.add_updater(lambda mob: mob.move_to(self.activation_and_axes.get_center()))
        if self.show_function_name and self.function_name is not None:
            activation_label = axes.get_graph_label(activation_graph, Text(self.function_name))
            always(activation_label.next_to, BoundingRec, DOWN)
        else:
            activation_label = None
        self.activation_mobject = VGroup(*[self.activation_and_axes, BoundingRec, activation_label])
        self.add(self.activation_mobject)

    def apply_activate_function(self, x):
        raise Exception("Activation Function Not Defined")


class ReLUActivation(ActivationFunction):
    def __init__(self):
        super().__init__(function_name='1. ReLU')

    def apply_activate_function(self, x):
        return x if x > 0 else 0


class SigmoidActivation(ActivationFunction):
    def __init__(self):
        super().__init__(function_name='2. Sigmoid',
                         x_range=[-10, 10],
                         y_range=[0, 1])

    def apply_activate_function(self, x):
        return 1 / (1 + math.exp(-x))


class TanhActivation(ActivationFunction):
    def __init__(self):
        super().__init__(function_name='3. Tanh',
                         x_range=[-10, 10])

    def apply_activate_function(self, x):
        return (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)


class TestActivation(Scene):
    def construct(self) -> None:
        activation = TanhActivation().scale(0.25)
        self.add(activation)
        self.wait()


def train_one_loop(model, optimizer, X, y):
    model.to(device=device)
    X.to(device=device, dtype=dtype)
    y.to(device=device, dtype=torch.long)

    scores = model(X)

    criterion = F.cross_entropy

    loss = criterion(scores, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

