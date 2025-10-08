I will describe the  interface you will use to solve word problems in ratio and proportions. The interface requires you to complete different steps to solve the problem. The order and choices you make to attempt to complete the steps is important and will demonstrate your thinking process. 


Interaction format
You interact with the ITS through an action that attempts a step (indicated by STEPNAME) or a request for hint. There are different levels of hints that can be requested indicated by "hint-level". In some special cases, if your attempt to complete a step is incorrect, the tutor may provide an automatic feedback indicated as "Tutor Intervention". In each screen, you can complete the steps in any order you want. You can perform as many attempts on a step as needed until the tutor says you have completed the step correctly. There is a special hint/intervention denoted by STRATEGYMESSAGE that you need to pay attention to since you will analyze its effect. 

Skills
Note that you will encounter problems one after another and your skills associated with the steps will be updated based on your performance. A subset of the steps are associated with skills. These skills get updated using a Bayesian Knowledge Tracing Model. Note that if you make errors in a step and then correct them, use hints (other than hints at level 1) to complete the step or need tutor interventions, the skill does not get updated for that step.


Interface

Screen 1
STEPNAME  PercentChangeType
A dropdown box with the following two options from which you must choose one.
Option 1: This is a percentage increase problem
Option 2: This is a percentage decrease problem
Skills used: “Identify percent change as increase or decrease”

Hint Requests
If hint-level1 is requested then display “Indicate whether the change in this problem was an increase or a decrease.”
If hint-level2 is requested then display “You were given the initial and final amounts. Did the amount increase or decrease from initial to final?”
If hint-level3 is requested then display the correct answer to this step
If there is an error in completing the step then display “You were given the initial and final amounts. Did the amount increase or decrease from initial to final?”


Proceed to Screen 2 after the step is completed correctly.


Screen 2
Complete three steps in the following proportion.  STEPNAMES: PercentageChange, AmountChange, AmountOriginal
Represent the unknown by a variable name.
PercentageChange/100 = AmountChange/AmountOriginal
Skills used
“Identify given amount of change in proportion”
“Identify given original amount in proportion”
“Represent amount of change in proportion with variable”
“Identify given percent change in proportion”
“Represent percent change in proportion with variable”


Hint Requests
If hint-level1 is requested at the PercentageChange step display “Enter the percent change in the problem.”
If hint-level2 is requested at the PercentageChange display the answer.
If hint-level1 is requested at the AmountChange step: “Enter the amount change in the problem.”
If hint-level2 is requested at the AmountChange display the answer.
If hint-level1 is requested at the AmountOriginal step: “Enter the original amount in the problem.”
If hint-level2 is requested at the AmountOriginal display the answer.

Tutor Intervention
If  the value entered is correct for one of the three steps but was mistakenly entered at the wrong step then display “The variable (or value) belongs elsewhere in the proportion.”


Proceed to Screen 3 after all steps were correctly completed


Screen 3

Three tabs are shown.
Tab 1 
Optional Task
I want to use Equivalent Ratios (you can checkmark this option). If you choose to do this, then you need to complete the following steps.
STEPNAMES: NF,DF, EA,UK 
Complete the steps in one of the following equations.
EA/100 = ((AmountChange*NF)/(AmountOriginal*DF))

EA/100 = ((AmountChange/NF)/(AmountOriginal/DF))

((EA*NF)/(100*DF)) = AmountChange/AmountOriginal

((EA/NF)/(100/DF)) = AmountChange/AmountOriginal

The unknown variable’s value is UK

Tab 2
Optional Task
I want to use Means and Extremes (you can checkmark this option). If you choose to do this, then you need to complete the following steps.
STEPNAMES: Mean1, Mean2, Extreme1, Extreme2, Prod, UK.
 Mean1 * Mean2 = Extreme1*Extreme2
If the left hand side contains the known variables
Prod=Extreme1*Extreme2
If the right hand side contains the known variables
Mean1 * Mean2 = Prod
The unknown variable’s value is UK


Tab 3
There are two steps to complete.
STEPNAMES: FA, UNITS
Complete the final answer denoted as FA and select an option from the dropdown for UNITS that contains the following options (final amount units, percentage increase, percentage decrease)
Skills used
“Calculate final amount using percent change from verbal problem statement”
“Calculate percent change from verbal problem statement”
“Enter label of final answer”

Hint Requests
If hint-level1 is requested at FA before completing optional steps in Tab1 or Tab2 display one of the following
For problems that ask for percentage change display “Enter the percent change in this scenario.”
For problems that ask for final amount display “Enter the final amount in this scenario.”

STRATEGYMESSAGE: If hint-level2 is requested at FA before completing optional steps in Tab1 or Tab2 display “If you don't know how to do that, try completing the optional steps for one of the two solution methods first. When the larger given denominator is an integer multiple of the smaller given denominator, the Equivalent Ratios method is usually easier than the Means and Extremes method because you wouldn't need to multiply by fractional or decimal values.”



If hint-level1 is requested at FA after completing optional steps in Tab1 or Tab2, display one of the following messages depending on whether the problem asks for percentage change or final amount.
For percentage change problems, display “You used the optional steps to determine the percent change; what value did you determine?”
For problems asking for the final amount, display “You used the optional steps to determine the amount of change; add or subtract that from the initial amount.”
If hint-level2 is requested at FA display the correct answer for this step.


If hint-level1 is requested for NF or DF steps, display “Enter the factor that will make the denominators of the ratios equal.”

If hint-level2 is requested for NF or DF , then display “Compare the denominators on each side of the equals sign. By what factor would you multiply/divide one denominator to make it equal to the other denominator?”

If hint-level2 is requested for NF after correctly completing DF, display “To maintain the equivalence of the two ratios, you must divide the numerator and denominator of this ratio by the same factor. By what factor did you divide the denominator?”

If hint-level3 is requested in NF or DF steps, display the correct answer for these steps.

If hint-level1 is requested at Mean1, Mean2, Extreme1, Extreme2, display “Set the product of the means equal to the product of the extremes.”
If hint-level2 is requested at Mean1, Mean2, Extreme1, Extreme2 display “Referring to the means and extremes of the above proportion, set the product of the denominator of the first ratio with the numerator of the other ratio to be equal to the denominator of the second ratio with the numerator of the first ratio.”
If hint-level3 is requested at Mean1, Mean2, Extreme1, Extreme2, display the correct answers for these steps.

If hint-level1 is requested at Prod, display “Enter the product of the two known quantities that are on the same side of the equation.”
If hint-level2 is requested at Prod, display the two known quantities and the message “What is their product?”
If hint-level3 is requested in Prod, display the correct answer

If hint-level1 is requested in UK, display “Enter the value for the unknown variable”
If hint-level2 is requested in UK, display the product of the two known variables and ask them to divide this by the remaining known variable.
If hint-level2 is requested in UK display the answer to this step.

If a hint-level1 is requested at the Units step, display one of the following.
For percentage change problems, display “Indicate whether the change in this problem was an increase or a decrease.”
For final amount problems, display “Indicate whether this is the final amount of a percentage change.”
If a hint-level2 is requested at Units, display the final answer.

Tutor Intervention


If the value entered for FA is the value for the unknown variable but the problem asks for a final amount, display “That is the amount of the change. What is the final amount.”

STRATEGYMESSAGE: When an incorrect value is entered at either EA or DF or NF, and the correct value of NF is a float, or
When the value of NF is entered at EA and is incorrect
the system shows "You might want to try the *Means and Extremes* method instead. Because the larger denominator (100) is not an integer multiple of the smaller denominator, the math for the *Equivalent Ratios* method is more difficult."
 
Example Problems
 
Problem 1
Evita collects baseball cards and is trying to complete a special edition set of cards.  Last week, she collected 30 cards.  This week she collected 80% more cards than she collected last week.  How many cards did she collect this week?

Problem 2
The Italian restaurant near your house sold 25 mushroom pizzas last week and 16 mushroom pizzas this week. What is the percentage change in the number of pizzas sold from last week to this week.

Problem 3
Last season, your favorite hockey team won 40 games.  So far this season, your favorite hockey team has won 7 more games than last season.  What is the percent change in the number of games that your favorite team won from last season to this season?

Your task
Assume that a student attempts problems similar to the examples one at a time. Suppose the student obtains the STRATEGYMESSAGE in K different problems, what is your prediction for whether the student will follow the strategy outlined in that message in subsequent problems. Note that Means and extremes is a general strategy (but requires more computation) whereas equivalent ratios is efficient in special cases. Also, the STRATEGYMESSAGE is shown to a student only if they ask for a hint (at a certain step) or if they perform an error due to which the tutor shows the intervention. Can you provide some confidence for your prediction and assumptions you have made.
