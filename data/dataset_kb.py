  

dataset_kb = {
        "Titanic": """The Titanic dataset is a classic dataset often used in machine learning and statistical analysis. It contains information about passengers on the Titanic and whether they survived the shipwreck. The dataset is widely used for binary classification tasks (survived or not).\             The dataset attributes are as follows: PassengerId: A unique identifier for each passenger, such as 1, 2, 3, etc.\T            Survived (Target Variable): Indicates whether the passenger survived. Possible values are 0 (did not survive) and 1 (survived).\a            Pclass (Passenger Class): Indicates the socio-economic class of the passenger. The possible values are 1 (upper class), 2 (middle class), and 3 (lower class).\
            Sex: Gender of the passenger. The possible values are 'male' and 'female'.\
            Age: Age of the passenger in years. It includes numerical values, which can be fractional for children (e.g., 0.42 for 5 months old). Missing values are represented as NaN.\
            SibSp (Number of Siblings/Spouses Aboard): This indicates the number of siblings or spouses the passenger had aboard the Titanic. The values are numerical, such as 0, 1, 2, etc.\
            Parch (Number of Parents/Children Aboard): This indicates the number of parents or children the passenger had aboard the Titanic. The values are numerical, such as 0, 1, 2, etc.\
            Fare: The fare paid by the passenger. It is represented as numerical values, such as 7.25 or 71.83.\
            Embarked: The port where the passenger boarded the Titanic. The possible values are 'C' (Cherbourg), 'Q' (Queenstown), and 'S' (Southampton).
            """.strip(),
        "Adult Income": """
            The Adult dataset, also known as the "Census Income" dataset, is commonly used for machine learning and data mining tasks. It contains information about individuals from the 1994 U.S. Census and is often used for binary classification tasks (income >50K or <=50K).
            The dataset attributes are as follows: Age: The age of the individual. It is represented as numerical values.\
            Workclass: The type of employment of the individual. Possible values include 'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', and 'Never-worked'.\
            fnlwgt: The final weight, which is a numeric value representing the number of people the census entry represents.\
            Education: The highest level of education achieved by the individual. Possible values include 'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', and '5th-6th'.\
            Education-num: The number of years of education completed by the individual. It is represented as numerical values.\
            Marital-status: The marital status of the individual. Possible values include 'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', and 'Married-AF-spouse'.\
            Occupation: The occupation of the individual. Possible values include 'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', and 'Armed-Forces'.\
            Relationship: The relationship of the individual to the household. Possible values include 'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', and 'Unmarried'.\
            Race: The race of the individual. Possible values include 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', and 'Black'.\
            Sex: The gender of the individual. Possible values are 'Male' and 'Female'.\
            Capital-gain: The capital gains of the individual. It is represented as numerical values.\
            Capital-loss: The capital losses of the individual. It is represented as numerical values.\
            Hours-per-week: The number of hours worked per week by the individual. It is represented as numerical values.\
            Native-country: The native country of the individual. Possible values include 'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'.\
            Income (Target Variable): Indicates whether the individual's income is greater than 50K or less than or equal to 50K. Possible values are 1 if income >50K and 0 if income <=50K.
            """.strip()
}