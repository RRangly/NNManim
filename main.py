from manim import *
from manim_slides import Slide, ThreeDSlide
import numpy as np
import pandas as pd
import math

np.random.seed(42)
class beginning(Slide):
    def construct(self):
        # Display MNIST Number
        self.next_slide()
        df = pd.read_csv("train.csv")
        displayNum = df.drop(df.columns[0], axis = 1).loc[6].tolist()
        pxsize = 0.2
        imgSL = 28
        pixels = VGroup()
        for i in range(imgSL):
            for j in range(imgSL):
                greyScale = displayNum[i*imgSL + j]
                hexVal = f"#{greyScale:02X}{greyScale:02X}{greyScale:02X}"
                pixel = Square(side_length=pxsize,stroke_width=0).set_fill(hexVal,opacity=1).move_to([(j)*pxsize,(-i)*pxsize,0])
                pixels.add(pixel)
        transformations = []
        pixels.move_to(ORIGIN)
        for i, pixel in enumerate(pixels):
            
            num = MathTex(f"{displayNum[i]/255:.3f}", font_size=8, color=WHITE)
            transformations.append(Transform(pixel, num.move_to([((i % (imgSL * 2))- imgSL)*pxsize ,(imgSL/2-(i // (imgSL * 2)))*pxsize,0])))
        self.play(Write(pixels))
        self.wait(0.2)
        self.next_slide()

        # Transform to numbers display of MNIST
        self.play(transformations)
        self.wait(0.2)
        self.next_slide()

        # Blank
        self.play(FadeOut(pixels))
        self.wait(0.2)
        self.next_slide()

class network(Slide):
    def construct(self):
        # Neural network - Input Layer
        inputLayerSize = 16
        hiddenLayerSize = 12
        inputLayer = VGroup()
        neuronSize = 0.1
        neuronDistance = 0.4
        inputLayerX = -3
        hiddenLayer1X = -1
        hiddenlayer2X = 1
        hiddenlayer3X = 3
        for i in range(inputLayerSize//2):
            inputLayer.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([inputLayerX,(i + 1.5)*neuronDistance,0]))
        for i in range(inputLayerSize//2):
            inputLayer.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([inputLayerX,-(i + 1.5)*neuronDistance,0]))
        self.play(Write(inputLayer))
        self.play(Write(MathTex("784", font_size=52, color=WHITE).move_to([inputLayerX,0,0])))
        self.wait(0.2)
        self.next_slide()

        # Hidden Layers
        hiddenlayer1 = VGroup()
        for i in range(hiddenLayerSize//2):
            hiddenlayer1.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([hiddenLayer1X,(i + 1.5)*neuronDistance,0]))
        for i in range(hiddenLayerSize//2):
            hiddenlayer1.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([hiddenLayer1X,-(i + 1.5)*neuronDistance,0]))
        self.play(Write(hiddenlayer1))
        self.play(Write(MathTex("64", font_size=52, color=WHITE).move_to([hiddenLayer1X,0,0])))
        lines1 = VGroup()
        for neuron in inputLayer:
            for neuron2 in hiddenlayer1:
                line = Line(neuron.get_center(), neuron2.get_center(), stroke_width=0.5)
                line.z_index = -1
                lines1.add(line)
        self.play(Write(lines1))
        hiddenlayer2 = VGroup()
        for i in range(hiddenLayerSize//2):
            hiddenlayer2.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([hiddenlayer2X,(i + 1)*neuronDistance,0]))
        for i in range(hiddenLayerSize//2):
            hiddenlayer2.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([hiddenlayer2X,-(i + 1)*neuronDistance,0]))
        self.play(Write(hiddenlayer2))
        self.play(Write(MathTex("32", font_size=52, color=WHITE).move_to([hiddenlayer2X,0,0])))
        lines2 = VGroup()
        for neuron in hiddenlayer1:
            for neuron2 in hiddenlayer2:
                line = Line(neuron.get_center(), neuron2.get_center(), stroke_width=0.5)
                line.z_index = -2
                lines2.add(line)
        self.play(Write(lines2))
        hiddenlayer3 = VGroup()
        for i in range(hiddenLayerSize//2):
            hiddenlayer3.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([hiddenlayer3X,(i + 1)*neuronDistance,0]))
        for i in range(hiddenLayerSize//2):
            hiddenlayer3.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([hiddenlayer3X,-(i + 1)*neuronDistance,0]))
        self.play(Write(hiddenlayer3))
        self.play(Write(MathTex("16", font_size=52, color=WHITE).move_to([hiddenlayer3X,0,0])))
        lines3 = VGroup()
        for neuron in hiddenlayer2:
            for neuron2 in hiddenlayer3:
                line = Line(neuron.get_center(), neuron2.get_center(), stroke_width=0.5)
                line.z_index = -1
                lines3.add(line)
        self.play(Write(lines3))
        self.wait(0.2)
        self.next_slide()

        # Output Layer
        outputLayerSize = 10
        outputLayerX = 5
        outputLayer = VGroup()
        outputTexts = VGroup()
        for i in range(outputLayerSize//2):
            outputLayer.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([outputLayerX,(i + 0.5)*neuronDistance,0]))
            outputTexts.add(MathTex(str(math.floor(5 - i)), font_size=24, color=WHITE).move_to([outputLayerX + neuronDistance,(i + 0.5)*neuronDistance,0]))
        for i in range(outputLayerSize//2):
            outputLayer.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([outputLayerX,-(i + 0.5)*neuronDistance,0]))
            outputTexts.add(MathTex(str(math.floor(6 + i)), font_size=24, color=WHITE).move_to([outputLayerX + neuronDistance,-(i + 0.5)*neuronDistance,0]))
        self.play(Write(outputLayer))
        lines4 = VGroup()
        for neuron in hiddenlayer3:
            for neuron2 in outputLayer:
                line = Line(neuron.get_center(), neuron2.get_center(), stroke_width=0.5)
                line.z_index = -1
                lines4.add(line)
        self.play(Write(lines4))
        self.add(outputTexts)
        self.wait(0.2)
        self.next_slide()

        # Example

        img = ImageMobject("7img.png").scale(1).move_to([-5,0,0])
        img.z_index = -1
        self.play(FadeIn(img))
        self.wait(0.2)
        self.next_slide()

        # forward propogation
        squares = VGroup()
        transformations2 = []
        neuronlit = np.random.rand(16)
        for i in range(inputLayerSize):
            square = Square(side_length=(neuronSize*2), color = WHITE, fill_opacity = 1, stroke_width=0).move_to(img.get_center())
            squares.add(square)
            trans = Transform(square, inputLayer[i].copy().set_fill(color=GREEN,opacity=neuronlit[i]))
            transformations2.append(trans)
        self.add(squares)
        self.play(transformations2)

        act1 = VGroup()
        hlLit1 = np.random.rand(12)
        for i in range(hiddenLayerSize):
            act1.add(hiddenlayer1[i].copy().set_fill(color=GREEN,opacity=hlLit1[i]))
        self.play(FadeIn(act1))

        act2 = VGroup()
        hlLit2 = np.random.rand(12)
        for i in range(hiddenLayerSize):
            act2.add(hiddenlayer2[i].copy().set_fill(color=GREEN,opacity=hlLit2[i]))
        self.play(FadeIn(act2))

        act3 = VGroup()
        hlLit3 = np.random.rand(12)
        for i in range(hiddenLayerSize):
            act3.add(hiddenlayer3[i].copy().set_fill(color=GREEN,opacity=hlLit3[i]))
        self.play(FadeIn(act3))

        act4 = outputLayer[3].copy().set_fill(color=GREEN,opacity=1)
        self.play(FadeIn(act4))
        self.next_slide()

class equations(Slide):
    def construct(self):
        #Equations
        self.clear()
        self.wait(0.2)
        equation = MathTex(
            r"a = \text{ReLU}\left(\sum_{i=1}^{n} w_i x_i + b\right)",
            r"= \max\left(0, \sum_{i=1}^{n} w_i x_i + b\right)",
            font_size=32, color=WHITE).move_to([0,2,0])
        equation2 = MathTex(
            r"a^{[l]} = f\left(W^{[l]}a^{[l-1]} + b^{[l]}\right)",
            font_size=32, color=WHITE).move_to([0,1,0])
        equation3 = MathTex(
            r"\sigma(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}",
            font_size=32, color=WHITE).move_to([0,0,0])
        equation4 = MathTex(
            r"L = \frac{1}{n}\sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2",
            font_size=32, color=WHITE).move_to([0,-1,0])
        equation5 = MathTex(
            r"\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}",
            font_size=32, color=WHITE).move_to([0,-2,0])
        self.play(Write(equation))
        self.play(Write(equation2))
        self.play(Write(equation3))
        self.play(Write(equation4))
        self.play(Write(equation5))
        self.wait(0.2)
        self.next_slide()


def natural_terrain(x, y):
    return (
        0.5 * np.sin(0.5 * x) * np.cos(0.5 * y)
        + 0.3 * np.sin(0.3 * x + 0.4 * y)
        + 0.2 * np.cos(0.2 * x - 0.3 * y)
    )
class gradientDescent(ThreeDSlide):
    def construct(self):
        self.clear()

        axes = ThreeDAxes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            z_range=[-4, 4, 0.5],)
        self.add(axes)
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.wait(0.2)
        self.next_slide()

        def rough_terrain(x, y):
            return (
                0.5 * np.sin(0.5 * x) * np.cos(0.5 * y)
                + 0.3 * np.sin(0.3 * x + 0.4 * y)
                + 0.2 * np.cos(0.2 * x - 0.3 * y)
                + 0.2 * np.sin(3 * x) * np.cos(3 * y)   # extra roughness term
                + 0.1 * np.sin(5 * x) * np.sin(5 * y)   # even higher frequency detail
            )
        
        # Define the gradient of the natural terrain function
        def grad_rough_terrain(x, y):
            dfdx = (
                0.25 * np.cos(0.5 * x) * np.cos(0.5 * y)
                + 0.09 * np.cos(0.3 * x + 0.4 * y)
                - 0.04 * np.sin(0.2 * x - 0.3 * y)
                + 0.6 * np.cos(3 * x) * np.cos(3 * y)
                + 0.5 * np.cos(5 * x) * np.sin(5 * y)
            )
            dfdy = (
                -0.25 * np.sin(0.5 * x) * np.sin(0.5 * y)
                + 0.12 * np.cos(0.3 * x + 0.4 * y)
                + 0.06 * np.sin(0.2 * x - 0.3 * y)
                - 0.6 * np.sin(3 * x) * np.sin(3 * y)
                + 0.5 * np.sin(5 * x) * np.cos(5 * y)
            )
            return np.array([dfdx, dfdy])
        surface = Surface(
            lambda u, v: axes.c2p(u, v, natural_terrain(u, v)),
            u_range=[-8, 8],
            v_range=[-8, 8],
            resolution=(32, 32),
            fill_opacity=0.8,
            checkerboard_colors=[GREEN_D, GREEN_E],
        )
        self.play(Create(surface), run_time=3)
        self.wait(0.2)
        self.next_slide()

        #Gradient Descent
        start = np.array([0, 0])
        start_point_3d = axes.c2p(start[0], start[1], rough_terrain(start[0], start[1]))
        dot = Dot3D(point=start_point_3d, color=YELLOW)
        self.play(Create(dot))
        num_steps = 500
        points = [start]
        for i in range(num_steps):
            current = points[i]
            grad = grad_rough_terrain(current[0], current[1])
            learning_rate = np.random.uniform(0.2, 0.3)
            next_point = current - (learning_rate * grad)
            points.append(next_point)
        print("pointsSize", len(points))
        path_points = [
            axes.c2p(pt[0], pt[1], rough_terrain(pt[0], pt[1])) for pt in points
        ]
        print("pathSize", len(path_points))
        path = VMobject()
        path.set_points_as_corners(path_points)
        path.stroke_n_points = len(path_points)
        self.play(Create(path), run_time=3)
        self.play(MoveAlongPath(dot, path), run_time=3, rate_func=linear)
        self.wait(0.2)
        self.next_slide()

class network2(Slide):
    def construct(self):
        # Neural network - Input Layer
        inputLayerSize = 16
        hiddenLayerSize = 12
        inputLayer = VGroup()
        neuronSize = 0.1
        neuronDistance = 0.4
        inputLayerX = -3
        hiddenLayer1X = -1
        hiddenlayer2X = 1
        hiddenlayer3X = 3
        for i in range(inputLayerSize//2):
            inputLayer.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([inputLayerX,(i + 1.5)*neuronDistance,0]))
        for i in range(inputLayerSize//2):
            inputLayer.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([inputLayerX,-(i + 1.5)*neuronDistance,0]))
        self.play(Write(inputLayer))
        self.play(Write(MathTex("784", font_size=52, color=WHITE).move_to([inputLayerX,0,0])))
        self.wait(0.2)
        self.next_slide()

        # Hidden Layers
        hiddenlayer1 = VGroup()
        for i in range(hiddenLayerSize//2):
            hiddenlayer1.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([hiddenLayer1X,(i + 1.5)*neuronDistance,0]))
        for i in range(hiddenLayerSize//2):
            hiddenlayer1.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([hiddenLayer1X,-(i + 1.5)*neuronDistance,0]))
        self.play(Write(hiddenlayer1))
        self.play(Write(MathTex("64", font_size=52, color=WHITE).move_to([hiddenLayer1X,0,0])))
        lines1 = VGroup()
        for neuron in inputLayer:
            for neuron2 in hiddenlayer1:
                line = Line(neuron.get_center(), neuron2.get_center(), stroke_width=0.5)
                line.z_index = -1
                lines1.add(line)
        self.play(Write(lines1))
        hiddenlayer2 = VGroup()
        for i in range(hiddenLayerSize//2):
            hiddenlayer2.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([hiddenlayer2X,(i + 1)*neuronDistance,0]))
        for i in range(hiddenLayerSize//2):
            hiddenlayer2.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([hiddenlayer2X,-(i + 1)*neuronDistance,0]))
        self.play(Write(hiddenlayer2))
        self.play(Write(MathTex("32", font_size=52, color=WHITE).move_to([hiddenlayer2X,0,0])))
        lines2 = VGroup()
        for neuron in hiddenlayer1:
            for neuron2 in hiddenlayer2:
                line = Line(neuron.get_center(), neuron2.get_center(), stroke_width=0.5)
                line.z_index = -2
                lines2.add(line)
        self.play(Write(lines2))
        hiddenlayer3 = VGroup()
        for i in range(hiddenLayerSize//2):
            hiddenlayer3.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([hiddenlayer3X,(i + 1)*neuronDistance,0]))
        for i in range(hiddenLayerSize//2):
            hiddenlayer3.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([hiddenlayer3X,-(i + 1)*neuronDistance,0]))
        self.play(Write(hiddenlayer3))
        self.play(Write(MathTex("16", font_size=52, color=WHITE).move_to([hiddenlayer3X,0,0])))
        lines3 = VGroup()
        for neuron in hiddenlayer2:
            for neuron2 in hiddenlayer3:
                line = Line(neuron.get_center(), neuron2.get_center(), stroke_width=0.5)
                line.z_index = -1
                lines3.add(line)
        self.play(Write(lines3))
        self.wait(0.2)
        self.next_slide()

        # Output Layer
        outputLayerSize = 10
        outputLayerX = 5
        outputLayer = VGroup()
        outputTexts = VGroup()
        for i in range(outputLayerSize//2):
            outputLayer.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([outputLayerX,(i + 0.5)*neuronDistance,0]))
            outputTexts.add(MathTex(str(math.floor(5 - i)), font_size=24, color=WHITE).move_to([outputLayerX + neuronDistance,(i + 0.5)*neuronDistance,0]))
        for i in range(outputLayerSize//2):
            outputLayer.add(Dot(radius=neuronSize, stroke_width=1, fill_opacity=1,fill_color=BLACK,stroke_color=WHITE).move_to([outputLayerX,-(i + 0.5)*neuronDistance,0]))
            outputTexts.add(MathTex(str(math.floor(6 + i)), font_size=24, color=WHITE).move_to([outputLayerX + neuronDistance,-(i + 0.5)*neuronDistance,0]))
        self.play(Write(outputLayer))
        lines4 = VGroup()
        for neuron in hiddenlayer3:
            for neuron2 in outputLayer:
                line = Line(neuron.get_center(), neuron2.get_center(), stroke_width=0.5)
                line.z_index = -1
                lines4.add(line)
        self.play(Write(lines4))
        self.add(outputTexts)
        self.wait(0.2)
        self.next_slide()

        # Example

        img = ImageMobject("7img.png").scale(1).move_to([-5,0,0])
        img.z_index = -1
        self.play(FadeIn(img))
        self.wait(0.2)
        self.next_slide()

        # forward propogation
        squares = VGroup()
        transformations2 = []
        neuronlit = np.random.rand(16)
        for i in range(inputLayerSize):
            square = Square(side_length=(neuronSize*2), color = WHITE, fill_opacity = 1, stroke_width=0).move_to(img.get_center())
            squares.add(square)
            trans = Transform(square, inputLayer[i].copy().set_fill(color=GREEN,opacity=neuronlit[i]))
            transformations2.append(trans)
        self.add(squares)
        self.play(transformations2)

        act1 = VGroup()
        hlLit1 = np.random.rand(12)
        for i in range(hiddenLayerSize):
            act1.add(hiddenlayer1[i].copy().set_fill(color=GREEN,opacity=hlLit1[i]))
        self.play(FadeIn(act1))

        act2 = VGroup()
        hlLit2 = np.random.rand(12)
        for i in range(hiddenLayerSize):
            act2.add(hiddenlayer2[i].copy().set_fill(color=GREEN,opacity=hlLit2[i]))
        self.play(FadeIn(act2))

        act3 = VGroup()
        hlLit3 = np.random.rand(12)
        for i in range(hiddenLayerSize):
            act3.add(hiddenlayer3[i].copy().set_fill(color=GREEN,opacity=hlLit3[i]))
        self.play(FadeIn(act3))

        act4 = outputLayer[6].copy().set_fill(color=GREEN,opacity=1)
        self.play(FadeIn(act4))
        self.next_slide()