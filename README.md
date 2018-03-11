# single_neuron_decision_boundary
The Purpose of this project is to practice with a single neuron decision boundary and learning rule. It will display a decision boundary of single neuron.

This program contains 3 sliders, 2 buttons, and one drop
down selection box.

• Sliders:-
      Slider 1: Changes w1 (first weight) between -10 and 10.
                Default value = 1
      Slider 2: Changes w2 (second weight) between -10 and 10.
                Default value = 1
      Slider 3: Changes b (bias between -10 and 10. Default
                value=0
      
• Buttons:-
      Button1: Train. Clicking this button adjusts the weights
                      and bias for 100 steps using the learning rule. Wnew =Wold+ epT
                      where e = t – a
      Button 2: Create random data. Assuming that there are only
                two possible target values 1 and -1 (two classes), this button
                creates 4 random data points (two points for each
                class). The range of data points should be from -10 to 10 for
                both dimensions.
                
                
• Drop Down Selection:-
      The drop down box allows the user to select between
      three transfer functions (Symmetrical Hard limit, Hyperbolic
      Tangent, and Linear)
