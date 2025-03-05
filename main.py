from manim import *
from manim_slides import Slide
import pandas as pd
import math
class beginning(Slide):
    def construct(self):
        # Display MNIST Number
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