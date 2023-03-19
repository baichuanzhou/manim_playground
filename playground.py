import torch.nn as nn
import torch.optim as optim
import torch
from manimlib import *
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ManimNeuralNet(VGroup):
    neuron_radius = 0.125
    neuron_to_neuron_buff = MED_SMALL_BUFF
    neuron_stroke_width = 1
    neuron_stroke_color = GREY
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
            ).set_fill(BLUE, opacity=0.7)
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
        return CubicBezier(
            neuron1.get_center() + [self.neuron_radius, 0, 0],
            neuron1.get_center() + [self.neuron_radius, 0, 0] + RIGHT,
            neuron2.get_center() - [self.neuron_radius, 0, 0] - RIGHT,
            neuron2.get_center() - [self.neuron_radius, 0, 0],
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
        self.NumberText = MTexText(f"{self.number.get_value()}").add_updater(
            lambda mob: mob.become(MTexText(f"{self.number.get_value()}").scale(0.5)))
        always(self.Text.next_to, self.NumberText, RIGHT)
        always(self.NumberText.next_to, self.Buttons, DOWN + LEFT, buff=SMALL_BUFF)
        super().__init__(value, self.NumberText, self.Text, self.Buttons, *args, **kwargs)
        self.add_mouse_press_listner(self.on_mouse_press)

    def set_value_anim(self, value):
        self.number.set_value(value)

    def toggle_value(self, point):
        plus_bounding = self.PlusButton.get_bounding_box()
        minus_bounding = self.MinusButton.get_bounding_box()

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


class Feature(ControlMobject):
    def __init__(self, text, init_value=False, *args, **kwargs):
        feature_text = MTexText(text)
        check_box = Checkbox(init_value)
        feature_text.add_updater(lambda mob: mob.next_to(check_box, LEFT))
        self.check_box = check_box
        super().__init__(init_value, feature_text, check_box, *args, **kwargs)


class Slider(ControlMobject):
    def __init__(self, text, min_value, max_value, step, init_value, num_decimal_places=2, *args, **kwargs):
        self.slider = LinearNumberSlider(min_value=min_value, max_value=max_value, step=step)
        self.slider.set_value(init_value)

        number = DecimalNumber(num_decimal_places=num_decimal_places).add_updater(
            lambda mob: mob.become(DecimalNumber(self.slider.get_value()))
            .next_to(self.slider, UP, buff=SMALL_BUFF).scale(0.3)
        )
        state_text = TexText(text, color=WHITE).add_updater(lambda mob: mob.next_to(number, UP))
        super().__init__(False, self.slider, number, state_text, *args, **kwargs)

    def get_value(self):
        return self.slider.get_value()


class NeuralNet(nn.Sequential):
    def __init__(self, layers_size, use_dropout=False, dropout=0.5, batchnorm=False, activation='ReLU'):
        super().__init__()
        self.layers_size = layers_size
        for i in range(len(layers_size) - 1):
            layer = nn.Sequential()

            layer.add_module(module=nn.Linear(layers_size[i], layers_size[i + 1]), name=f"Linear{i}")

            if activation == 'ReLU':
                layer.add_module(module=nn.ReLU(), name=f"{activation}{i}")
            elif activation == 'Sigmoid':
                layer.add_module(module=nn.Sigmoid(), name=f"{activation}{i}")
            else:
                layer.add_module(module=nn.Tanh(), name=f"{activation}{i}")

            if batchnorm:
                layer.add_module(module=nn.BatchNorm1d(layers_size[i + 1]), name=f"BatchNorm1d{i}")

            if use_dropout:
                layer.add_module(module=nn.Dropout(p=dropout), name=f"Dropout{i}")
            self.add_module(module=layer, name=f"LinearBlock{i}")
        self.add_module(module=nn.Linear(layers_size[len(layers_size) - 1], 2), name='Fully Connected')


class PlayGround(Scene):
    features = ['$x_1$', '$x_2$', '${x_1}^2$', '${x_2}^2$', '$x_1x_2$', '$sin(x_1)$', '$sin(x_2)$']

    def construct(self) -> None:
        self.create_button = MyButton('create').scale(0.3)
        self.create_button.to_edge(DL)
        self.add(self.create_button)
        self.hidden_layer_controls = VGroup()
        for i in range(6):
            self.hidden_layer_controls.add(HiddenLayerControl(max_num=8).scale(0.9))

        self.features_box = VGroup()

        for feature in self.features:
            init_value = (feature == '$x_1$' or feature == '$x_2$')
            self.features_box.add(Feature(feature, init_value).scale(0.7))
        self.features_box.arrange(RIGHT)
        self.features_box.next_to(self.create_button, RIGHT)
        self.add(self.features_box)

        self.hidden_layer_controls.arrange(DOWN)
        self.hidden_layer_controls.next_to(self.create_button, UP)
        self.add(self.hidden_layer_controls)

        self.nn = ManimNeuralNet([])
        self.layers_size = []
        self.learning_rate_slider = Slider(
            'learning rate', 0, 1, 0.01, 0.01, num_decimal_places=3
        ).scale(0.4)
        self.dropout_slider = Slider('dropout', 0, 1, 0.1, 1).scale(0.4)
        self.batch_size_slider = Slider('batch size', 20, 100, 1, 20).scale(0.4)
        self.sliders = Group(*[self.learning_rate_slider, self.dropout_slider, self.batch_size_slider])
        self.sliders.arrange(RIGHT)
        self.sliders.to_edge(UP)
        self.add(self.sliders)
        self.learning_rate = self.learning_rate_slider.get_value()
        self.dropout = self.dropout_slider.get_value()
        self.batch_size = self.batch_size_slider.get_value()

        self.ax = Axes(
            x_range=[0, 100],
            y_range=[0, 10],
            axis_config={'include_numbers': False}
        ).scale(0.2).to_edge(UR)
        self.graph = self.ax.get_graph(
            lambda x: np.random.normal(size=(1, 200))[0][-101:][int(np.floor(x))] + 4).add_updater(
            lambda mob: mob.become(
                self.ax.get_graph(lambda x: np.random.normal(size=(1, 200))[0][-101:][int(np.floor(x))] + 4)
            )
        )
        self.dots = VGroup()
        self.add(self.dots)
        self.datasets = VGroup(*[DataSet(get_data_spiral, noise=0.2),
                                 DataSet(get_data_gaussian, noise=1),
                                 DataSet(get_data_circle, noise=0.1),
                                 DataSet(get_data_xor, noise=0.1)
                                 ])
        self.datasets.arrange(RIGHT)
        self.datasets.to_edge(UL).move_to(self.datasets.get_center() + LEFT * 0.3)
        self.add(self.datasets)

        self.show_datasets = VGroup(*[DataSet(get_data_spiral, noise=0.2),
                                 DataSet(get_data_gaussian, noise=1),
                                 DataSet(get_data_circle, noise=0.1),
                                 DataSet(get_data_xor, noise=0.1)
                                 ])

        for dataset in self.show_datasets:
            dataset.scale(2.5).center().to_edge(RIGHT)

        self.dataset = self.show_datasets[0]
        # self.dataset.center()
        # self.dataset.to_edge(RIGHT)
        self.add(self.dataset)

        # self.loss = ValueTracker(0)

        # self.loss_text = TexText(f'Loss: {self.loss.get_value()}').add_updater(
        #     lambda mob: mob.become(
        #         TexText(f'Loss: {self.loss.get_value()}')
        #     ).next_to(self.dataset, UP).scale(0.5).set_color(WHITE)
        # )

    def check_pressed(self, mob: Mobject, point: np.ndarray):
        bounding_box = mob.get_bounding_box()
        return bounding_box[0][0] < point[0] < bounding_box[2][0] and \
            bounding_box[0][1] < point[1] < bounding_box[2][1]

    def press_datasets(self, point: np.ndarray):
        if self.check_pressed(self.datasets[0], point):
            return self.show_datasets[0]
        elif self.check_pressed(self.datasets[1], point):
            return self.show_datasets[1]
        elif self.check_pressed(self.datasets[2], point):
            return self.show_datasets[2]
        elif self.check_pressed(self.datasets[3], point):
            return self.show_datasets[3]
        return None

    def reset(self, point):
        self.remove(self.dataset)
        self.remove(self.nn)
        self.remove(self.ax)
        self.remove(self.graph)
        self.remove(self.dots)
        self.layers_size = []
        self.model = None
        self.optimizer = None
        self.stop = True
        # self.remove(self.loss_text)
        # self.loss.set_value(0)

        if self.press_datasets(point) and (self.press_datasets(point).name != self.dataset.name):
            self.dataset = self.press_datasets(point)
            self.add(self.dataset)
        else:
            self.add(self.dataset)

        if self.check_pressed(self.create_button, point):
            now_layers_size = [hidden_layer_control.number.get_value()
                               for hidden_layer_control in self.hidden_layer_controls
                               if hidden_layer_control.number.get_value() != 0]

            now_features = [feature for i, feature in enumerate(self.features)
                            if self.features_box[i].check_box.get_value()]
            self.input_features = now_features
            now_layers_size = [len(self.input_features)] + now_layers_size
            if now_layers_size != self.layers_size:
                self.layers_size = now_layers_size.copy()
                new_nn = ManimNeuralNet(self.layers_size + [2]).next_to(self.hidden_layer_controls, RIGHT,
                                                                        buff=MED_SMALL_BUFF)
                self.remove(self.nn)
                self.add(new_nn)
                self.nn = new_nn
                self.add(self.nn)
            self.learning_rate = self.learning_rate_slider.get_value()
            self.dropout = self.dropout_slider.get_value()
            self.batch_size = self.batch_size_slider.get_value()

            ######################################################################
            #
            #       START TRAINING HERE
            #
            ######################################################################
            print(self.dropout)
            self.model = NeuralNet(self.layers_size, dropout=self.dropout, use_dropout=True, batchnorm=True)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            self.xs = torch.linspace(-6, 6, steps=40)
            self.ys = torch.linspace(-6, 6, steps=40)

            self.xx, self.yy = torch.meshgrid(self.xs, self.ys, indexing='xy')
            self.x_in = torch.cat((self.xx.resize(1600, 1), self.yy.resize(1600, 1)), 1)
            self.xx = self.xx.resize(1600, 1)
            self.yy = self.yy.resize(1600, 1)

            self.all_preds = torch.randn(1600, 1)
            self.all_preds[self.all_preds > 0] = 1
            self.all_preds[self.all_preds < 0] = 0

            self.x_line_pos = self.dataset.x_number_line.n2p(self.x_in[:, 0])
            y_line_pos = self.dataset.y_number_line.n2p(self.x_in[:, 1])
            y_diff = y_line_pos[:, 1] - self.x_line_pos[:, 1]
            self.x_line_pos[:, 1] += y_diff

            self.colors = [RED, BLUE]

            self.dots = VGroup(*[Dot(pos, color=self.colors[int(self.all_preds[i])]).set_opacity(0.3).scale(0.9)
                                 for i, pos in enumerate(self.x_line_pos)])
            self.add(self.dots)
            self.stop = False
            # self.add(self.loss_text)
            self.train()

    def update_stop(self):
        return self.stop

    def train(self):
        X_numpy, y_numpy = self.dataset.X_numpy, self.dataset.y_numpy
        X_train, X_test, y_train, y_test = train_test_split(X_numpy, y_numpy, test_size=0.1)
        X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
        X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

        model = self.model  # NeuralNet([2, 8, 8, 8, 8, 8], batchnorm=True, use_dropout=False, dropout=0.2)
        optimizer = self.optimizer  # optim.Adam(model.parameters(), lr=0.01)

        criterion = F.cross_entropy

        for epoch in range(1000):
            if self.stop is not True:
                model.train()
                scores = model(X_train)

                loss = criterion(scores, y_train)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

                if epoch % 10 == 0 and epoch != 0:
                    model.eval()
                    self.remove(self.dots)
                    _, self.all_preds = model(self.x_in).max(1)

                    self.dots = VGroup(*[Dot(pos, color=self.colors[self.all_preds[i]]).set_opacity(0.3).scale(0.7)
                                            for i, pos in enumerate(self.x_line_pos)])
                    self.add(self.dots)

                    self.wait(0.001)
                    # self.loss.set_value(loss.item())
                    time.sleep(0.001)

            self.stop = self.update_stop()
            if self.stop is True:
                model.zero_grad()
                optimizer.zero_grad()
                break

    def need_reset(self, point: np.ndarray):
        return self.check_pressed(self.create_button, point) or self.learning_rate != \
            self.learning_rate_slider.get_value() or self.dropout != self.dropout_slider.get_value() or \
            self.batch_size != self.batch_size_slider.get_value() or (self.press_datasets(point) is not None and
                                                                      (self.press_datasets(point).name !=
                                                                       self.dataset.name))

    def on_mouse_press(
            self,
            point: np.ndarray,
            button: int,
            mods: int
    ) -> None:
        super().on_mouse_press(point, button, mods)
        print(self.need_reset(point))

        if self.need_reset(point):
            self.reset(point)


def get_data_spiral(n=200, d=2, c=2, std=0.2):
    with torch.no_grad():
        X = torch.zeros(n * c, d)
        y = torch.zeros(n * c, dtype=torch.long)
        for i in range(c):
            index = 0
            r = torch.linspace(0.5, 5, n)
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


def get_data_gaussian(n=200, d=2, c=2, std=1):
    with torch.no_grad():
        X = torch.zeros(n * c, d)
        y = torch.zeros(n * c, dtype=torch.long)
        sign = 1
        for i in range(c):
            for ix in range(n * i, n * (i + 1)):
                X[ix] = torch.FloatTensor(torch.normal(mean=2.5 * sign, std=std, size=(1, 2)))
                y[ix] = i
            sign *= -1

        return X.numpy(), y.numpy()


def get_data_circle(n=200, d=2, std=0.1):
    with torch.no_grad():
        X = torch.zeros(n * 2, d)
        y = torch.zeros(n * 2, dtype=torch.long)

        r = 5

        inside_radius = torch.distributions.Uniform(0, r * 0.5).sample((n, 1))
        inside_angle = torch.distributions.Uniform(0, 2 * torch.pi).sample((n, 1))
        inside_x = inside_radius * torch.sin(inside_angle)
        inside_y = inside_radius * torch.cos(inside_angle)
        noise_x = torch.distributions.Uniform(-r, +r).sample((n, 1)) * std
        noise_y = torch.distributions.Uniform(-r, +r).sample((n, 1)) * std
        inside_x += noise_x
        inside_y += noise_y

        X[:n, 0] = inside_x[:, 0]
        X[:n, 1] = inside_y[:, 0]
        y[:n] = 0

        outside_radius = torch.distributions.Uniform(r * 0.7, r).sample((n, 1))
        outside_angle = torch.distributions.Uniform(0, 2 * torch.pi).sample((n, 1))
        outside_x = outside_radius * torch.sin(outside_angle)
        outside_y = outside_radius * torch.cos(outside_angle)
        noise_x = torch.distributions.Uniform(-r, +r).sample((n, 1)) * std
        noise_y = torch.distributions.Uniform(-r, +r).sample((n, 1)) * std
        outside_x += noise_x
        outside_y += noise_y

        X[n:, 0] = outside_x[:, 0]
        X[n:, 1] = outside_y[:, 0]
        y[n:] = 1
        return X.numpy(), y.numpy()


def get_data_xor(n=200, d=2, std=0):
    with torch.no_grad():
        X = torch.zeros(n * 2, d)
        Y = torch.zeros(n * 2, dtype=torch.long)

        x = torch.distributions.Uniform(-5, 5).sample((n * 2, 1))
        y = torch.distributions.Uniform(-5, 5).sample((n * 2, 1))

        x_below_zero = x < 0
        y_below_zero = y > 0
        padding_x, padding_y = torch.ones_like(x), torch.ones_like(y)
        padding_x[x_below_zero] = -1
        padding_y[y_below_zero] = -1
        padding_x *= 0.3
        padding_y *= 0.3

        x += padding_x + torch.distributions.Uniform(-5, 5).sample((n * 2, 1)) * std
        y += padding_y + torch.distributions.Uniform(-5, 5).sample((n * 2, 1)) * std

        X[:, 0] = x[:, 0]
        X[:, 1] = y[:, 0]

        label = (X[:, 0] * X[:, 1] > 0)
        Y[label] = 1

        return X.numpy(), Y.numpy()


class DataSet(VGroup):
    def __init__(self, generate_func, noise=0.5, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        bounding_box = Square(1, stroke_width=1).set_color(WHITE)
        self.add(bounding_box)
        self.x_number_line = NumberLine(
            x_range=[-6, 6, 1],
            include_numbers=True,
            label_direction=UP
        ).scale(1 / 12)
        self.y_number_line = NumberLine(
            x_range=[-6, 6, 1],
            include_numbers=True
        ).rotate(DEGREES * 90).scale(1 / 12)

        self.x_number_line.move_to(bounding_box.get_bounding_box()[0] + RIGHT * 0.492 + DOWN * 0.02)
        self.y_number_line.move_to(bounding_box.get_bounding_box()[2] + DOWN * 0.508 + RIGHT * 0.02)

        self.generate_func = generate_func
        self.noise = noise

        self.add(self.x_number_line)
        self.add(self.y_number_line)
        self.X, self.y = self.add_datapoints()
        self.X_numpy, self.y_numpy = self.X.numpy(), self.y.numpy()

        self.name = generate_func.__name__

    def add_datapoints(self):
        X, y = self.generate_func(n=1000, std=self.noise)
        colors = [RED_C, BLUE_C, PURPLE, YELLOW, PINK]
        x_line_pos = self.x_number_line.n2p(X[:, 0])
        y_line_pos = self.y_number_line.n2p(X[:, 1])
        y_diff = y_line_pos[:, 1] - x_line_pos[:, 1]
        x_line_pos[:, 1] += y_diff

        dots = VGroup(*[Dot(pos, color=colors[y[i]]).scale(0.05)
                        for i, pos in enumerate(x_line_pos)])
        self.add(dots)
        return torch.from_numpy(X), torch.from_numpy(y)


def optimizer_to(optim, device=device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


class ShowTraining(Scene):
    def construct(self) -> None:
        self.dataset = DataSet(get_data_spiral, noise=0.1).scale(2.5)
        self.add(self.dataset)
        self.dots = VGroup()
        self.colors = [RED, BLUE]

        self.xs = torch.linspace(-6, 6, steps=40)
        self.ys = torch.linspace(-6, 6, steps=40)

        self.xx, self.yy = torch.meshgrid(self.xs, self.ys, indexing='xy')
        self.x_in = torch.cat((self.xx.resize(1600, 1), self.yy.resize(1600, 1)), 1)
        self.xx = self.xx.resize(1600, 1)
        self.yy = self.yy.resize(1600, 1)

        self.all_preds = torch.randn(1600, 1)
        self.all_preds[self.all_preds > 0] = 1
        self.all_preds[self.all_preds < 0] = 0

        self.x_line_pos = self.dataset.x_number_line.n2p(self.x_in[:, 0])
        y_line_pos = self.dataset.y_number_line.n2p(self.x_in[:, 1])
        y_diff = y_line_pos[:, 1] - self.x_line_pos[:, 1]
        self.x_line_pos[:, 1] += y_diff

        self.dots = VGroup(*[Dot(pos, color=self.colors[int(self.all_preds[i])]).set_opacity(0.3).scale(0.7)
                             for i, pos in enumerate(self.x_line_pos)])
        self.add(self.dots)

        self.train()

    def train(self):
        X_numpy, y_numpy = self.dataset.X_numpy, self.dataset.y_numpy
        X_train, X_test, y_train, y_test = train_test_split(X_numpy, y_numpy, test_size=0.1)
        X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
        X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

        model = NeuralNet([2, 8, 8, 8, 8, 8], batchnorm=True, use_dropout=False, dropout=0.2)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        criterion = F.cross_entropy

        for epoch in range(10000):
            model.train()
            scores = model(X_train)

            loss = criterion(scores, y_train)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            print(loss.item())

            if epoch % 10 == 0 and epoch != 0:
                model.eval()
                self.remove(self.dots)
                _, self.all_preds = model(self.x_in).max(1)

                self.dots = VGroup(*[Dot(pos, color=self.colors[self.all_preds[i]]).set_opacity(0.3).scale(0.7)
                                        for i, pos in enumerate(self.x_line_pos)])
                self.add(self.dots)

                self.wait(0.01)
                time.sleep(0.001)


