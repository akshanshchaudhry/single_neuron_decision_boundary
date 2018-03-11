
import tkinter as tk
import numpy as np
from tkinter import simpledialog
from tkinter import filedialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import random as rnd
import math



class WidgetsWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.center_frame = tk.Frame(self)
        # Create a frame for plotting graphs
        self.left_frame = DisplayActivationFunctions(self, self.center_frame, bg='blue')


        self.center_frame.grid(row=1, column=0,sticky=tk.N + tk.E + tk.S + tk.W)
        self.center_frame.grid_propagate(True)
        self.center_frame.rowconfigure(1, weight=1, uniform='xx')
        self.center_frame.columnconfigure(0, weight=1, uniform='xx')
        self.center_frame.columnconfigure(1, weight=1, uniform='xx')

        self.left_frame.grid(row=0, column=0,sticky=tk.N + tk.E + tk.S + tk.W)



class DisplayActivationFunctions(tk.Frame):
    """
    This class is for displaying activation functions for NN.

    """

    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.xmin = -11
        self.xmax = 11
        self.ymin = -11
        self.ymax = 11
        self.first_weight = 1
        self.second_weight = 1
        self.input_weight_one = 1
        self.input_weight_two = 1

        self.bias = 0

        self.activation_function= "Symmetrical Hard limit"

        self.target = [-1,-1,1,1]
        self.input = []
        self.inputtbu = []
        self.weights = []
        self.weights_plot = []



        # self.gamma= self.alpha
        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Input')
        self.axes.set_ylabel('Output')
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)

        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=3, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=5, uniform='xx')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='xx')

        # set up the sliders for first weight
        self.first_weight_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10 , to_=10 , resolution=1, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="First Weight",
                                            command=lambda event: self.first_weight_slider_callback())
        self.first_weight_slider.set(self.first_weight)
        self.first_weight_slider.bind("<ButtonRelease-1>", lambda event: self.first_weight_slider_callback())
        self.first_weight_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # set up the sliders for second weight
        self.second_weight_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_= -1, to_= 10, resolution=1, bg="#DDDDDD",
                                     activebackground="#FF0000",
                                     highlightcolor="#00FFFF",
                                     label="Second Weight",
                                     command=lambda event: self.second_weight_slider_callback())
        self.second_weight_slider.set(self.second_weight)
        self.second_weight_slider.bind("<ButtonRelease-1>", lambda event: self.second_weight_slider_callback())
        self.second_weight_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        # set up the sliders for bias
        self.bias_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                         from_= -10, to_=10, resolution=1, bg="#DDDDDD",
                                         activebackground="#FF0000",
                                         highlightcolor="#00FFFF",
                                         label="Bias",
                                         command=lambda event: self.bias_slider_callback())
        self.bias_slider.set(self.bias)
        self.bias_slider.bind("<ButtonRelease-1>", lambda event: self.bias_slider_callback())
        self.bias_slider.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)



        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='xx')

        self.label_for_activation_function = tk.Label(self.buttons_frame, text="Activation Function", justify="center")
        self.label_for_activation_function.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.buttons_frame, self.activation_function_variable,
                                                          "Hyperbolic Tangent", "Linear", "Symmetrical Hard limit",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Symmetrical Hard limit")
        self.activation_function_dropdown.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.adjust_weight_button= tk.Button(self.buttons_frame, text="Train", command = self.adjust_weight_button_callback)
        self.adjust_weight_button.grid(row=0,column=1,sticky=tk.N + tk.E + tk.S + tk.W)

        self.random_button = tk.Button(self.buttons_frame, text="Create Random Weigths", command = self.random_button_callback)
        self.random_button.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        ############################################
        #GUI ends here
        ############################################
        self.random_button_callback()
        self.weightsandinputcalc()
        self.display_activation_function()







    def adjust_weight_button_callback(self):

        self.weightsandinputcalc()

        self.neuron_calc()


#The calculation of neuron is done in neuron_calc function

    def neuron_calc(self):

        weight = self.weights


        for j in range (0, 100):
            for i in range (0, len(self.inputtbu)):
                temp_netvalue = np.dot(self.inputtbu[i], weight) + self.bias
                output = self.outputcalc(temp_netvalue)

                error = self.target[i] - output

                weight = weight + error * self.inputtbu[i].reshape(2,1)
                self.bias = self.bias + error

        self.weights_plot = weight
        self.input_weight_one = self.weights_plot[0]
        self.input_weight_two = self.weights_plot[1]
        self.display_activation_function()




    def weightsandinputcalc(self):

        self.first_weight = self.first_weight_slider.get()
        self.second_weight = self.second_weight_slider.get()
        self.weights = np.array([self.first_weight, self.second_weight])
        self.weights = self.weights.reshape(2,1)
        # print(self.weights.shape)


#calculation of all the activation functions

    def outputcalc(self,x):
        if (self.activation_function == "Symmetrical Hard limit"):
            if x >= 0:
                netvalue = 1
                return netvalue
            else:
                netvalue = -1
                return netvalue

        elif (self.activation_function == "Linear"):
            netvalue = x
            return netvalue

        elif (self.activation_function == "Hyperbolic Tangent"):
            netvalue = (math.exp(x) -  math.exp(-x))/(math.exp(x) + math.exp(-x))
            return netvalue


    def display_activation_function(self):

        self.axes.cla()
        plt.xlim(self.xmin,self.xmax)
        plt.ylim(self.ymin,self.ymax)
        plt.xlabel("x axis")
        plt.ylabel("y axis")


        self.axes.plot(self.input[0][:2],self.input[1][:2],'ro')
        self.axes.plot(self.input[0][2:], self.input[1][2:], 'bs')

        self.canvas.draw()



        x_plot = [-11,11]
        y_plot = [(10 * self.input_weight_one - self.bias)/self.input_weight_two,(-10 * self.input_weight_one - self.bias)/self.input_weight_two]
        self.axes.plot(x_plot, y_plot)

        self.canvas.draw()


    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()
        self.display_activation_function()

    def first_weight_slider_callback(self):
        self.first_weight = self.first_weight_slider.get()
        self.display_activation_function()

    def second_weight_slider_callback(self):
        self.second_weight = self.second_weight_slider.get()
        self.display_activation_function()


    def bias_slider_callback(self):
        self.bias = self.bias_slider.get()
        self.display_activation_function()



    def random_button_callback(self):
        xpoints_temp = []
        ypoints_temp = []
        for i in range(0, 4):
            x = rnd.randint(-10, 10)
            y = rnd.randint(-10, 10)
            xpoints_temp.append(x)
            ypoints_temp.append(y)

        self.input = np.array([xpoints_temp,ypoints_temp])
        self.inputtbu = np.transpose(self.input)


        self.axes.cla()
        self.display_activation_function()



def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

widgets_window = WidgetsWindow()
# widgets_window.geometry("500x500")
# widgets_window.wm_state('zoomed')
widgets_window.title('Decision Boundary')
widgets_window.minsize(600,300)
widgets_window.protocol("WM_DELETE_WINDOW", lambda root_window=widgets_window: close_window_callback(root_window))
widgets_window.mainloop()
