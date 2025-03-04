from manim import *

class NeuralNetworkExplanation(Scene):
    def construct(self):
        # Step 1: Display the MNIST image.
        # Ensure that "mnist.png" exists in your project folder.
        mnist_image = ImageMobject("mnist.png")
        mnist_image.scale(2)
        self.play(FadeIn(mnist_image))
        self.wait(2)

        # Step 2: Animate the flattening process.
        # Here we create a row of squares to represent a flattened input vector.
        # (For simplicity, we only create 20 squares as a representative subset.)
        squares = VGroup(*[Square(side_length=0.3) for _ in range(20)])
        squares.arrange(RIGHT, buff=0.1)
        squares.to_edge(DOWN)
        # Animate a transformation from the image (via a copy) to the row of squares.
        # self.play(ReplacementTransform(mnist_image.copy(), squares))
        self.play(FadeOut(mnist_image), FadeIn(squares))
        self.play(FadeOut(mnist_image))
        self.wait(1)

        # Step 3: Build a simple neural network diagram.
        # Create the Input, Hidden, and Output layers.
        # For clarity, we use only a few neurons for input and hidden layers.
        input_neurons = VGroup(*[Circle(radius=0.2) for _ in range(3)])
        input_neurons.arrange(DOWN, center=True, buff=0.5)
        input_neurons.to_edge(LEFT, buff=1)
        input_label = Text("Input").scale(0.6)
        input_label.next_to(input_neurons, UP, buff=0.3)

        hidden_neurons = VGroup(*[Circle(radius=0.2) for _ in range(3)])
        hidden_neurons.arrange(DOWN, center=True, buff=0.5)
        hidden_neurons.next_to(input_neurons, RIGHT, buff=1)
        hidden_label = Text("Hidden").scale(0.6)
        hidden_label.next_to(hidden_neurons, UP, buff=0.3)

        output_neurons = VGroup(*[Circle(radius=0.2) for _ in range(10)])
        output_neurons.arrange(DOWN, center=True, buff=0.3)
        output_neurons.to_edge(RIGHT, buff=1)
        output_label = Text("Output").scale(0.6)
        output_label.next_to(output_neurons, UP, buff=0.3)

        # Fade in the network layers and their labels.
        self.play(FadeIn(input_neurons), FadeIn(input_label))
        self.play(FadeIn(hidden_neurons), FadeIn(hidden_label))
        self.play(FadeIn(output_neurons), FadeIn(output_label))
        self.wait(1)

        # Draw connections (arrows) between layers.
        connections = VGroup()
        # Connect input to hidden.
        for in_neuron in input_neurons:
            for h_neuron in hidden_neurons:
                arrow = Arrow(start=in_neuron.get_right(), 
                              end=h_neuron.get_left(), 
                              buff=0.1, stroke_width=2)
                connections.add(arrow)
        # Connect hidden to output.
        for h_neuron in hidden_neurons:
            for out_neuron in output_neurons:
                arrow = Arrow(start=h_neuron.get_right(), 
                              end=out_neuron.get_left(), 
                              buff=0.1, stroke_width=2)
                connections.add(arrow)
        self.play(Create(connections))
        self.wait(1)

        # Step 4: Animate the propagation of data.
        # A dot will represent the data as it moves through the network.
        dot = Dot(point=input_neurons[0].get_center(), color=YELLOW)
        self.play(FadeIn(dot))
        self.wait(0.5)
        
        # Animate dot moving from an input neuron to a hidden neuron.
        path1 = Line(input_neurons[0].get_center(), hidden_neurons[1].get_center())
        self.play(MoveAlongPath(dot, path1), run_time=1)
        self.wait(0.5)

        # Animate dot moving from the hidden neuron to an output neuron.
        path2 = Line(hidden_neurons[1].get_center(), output_neurons[4].get_center())
        self.play(MoveAlongPath(dot, path2), run_time=1)
        self.wait(0.5)

        # Finally, highlight the output neuron and label it with a digit (e.g., "7").
        predicted_digit = Tex("7").scale(0.8)
        predicted_digit.move_to(output_neurons[4].get_center())
        self.play(ReplacementTransform(dot.copy(), predicted_digit))
        self.wait(2)
