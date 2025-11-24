  

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
            """.strip(),
        "California Housing": """
            The California Housing dataset is a well-known dataset commonly used in machine learning, especially for predictive modeling and classification tasks. It contains aggregated housing information from California districts and is widely used to study socio-economic factors and housing market patterns. 
            In this version of the dataset, the target variable is transformed into a binary label indicating whether the median house value in a block group is above (1) or below (0) the overall median value, making it suitable for binary classification problems.
            The dataset attributes are as follows:
            MedInc (Median Income): Represents the median income of households in the block group. This feature is numerical and scaled such that a value of 1 corresponds to $10,000 in median income (e.g., 3.5 ≈ $35,000).\
            HouseAge (Median House Age): Indicates the median age of the houses in the block group in years. The values are numerical, such as 5, 20, or 42.\
            AveRooms (Average Number of Rooms): Represents the average number of rooms per household in the block group. This is a numerical value obtained by dividing the total number of rooms by the number of households.\
            AveBedrms (Average Number of Bedrooms): Similar to AveRooms, this indicates the average number of bedrooms per household in the block group. It is represented as a numerical value.\
            Population: The total population of the block group. This attribute contains numerical values, such as 500, 1520, or 3500.\
            AveOccup (Average Household Occupancy): Indicates the average number of occupants per household in the block group. It is a numerical value representing household density.\
            Latitude: The geographical latitude of the block group within California. It is represented as numerical coordinates.\
            Longitude: The geographical longitude of the block group within California. It is also represented as numerical coordinates.\
            MedHouseVal (Target Variable): A binary label indicating whether the median house value for the block group is above (1) or below (0) the median of all house values in the dataset.
        """.strip(),
        "Diabetes": """The Diabetes dataset (LARS) is a clinical dataset commonly used in machine learning. It contains physiological data from 442 diabetes patients, and it focuses on blood serum measurements to solve a binary classification task.\
            The dataset attributes are as follows:\
            age: Age of the passenger in years.\
            sex: Biological sex of the patient. The values are 'Male' and 'Female'.\
            bmi (Body Mass Index): A measure of body fat based on height and weight. Values are raw clinical measurements (e.g., 26.9).\
            bp (Average Blood Pressure): Mean arterial pressure (MAP) measured in mmHg.\
            s1 (Total Serum Cholesterol): Total amount of cholesterol in the blood (TC).\
            s2 (Low-Density Lipoproteins): Often called 'bad cholesterol' (LDL).\
            s3 (High-Density Lipoproteins): Often called 'good cholesterol' (HDL).\
            s4 (Total Cholesterol / HDL Ratio): A calculated risk factor. Higher values indicate higher risk.\
            s5 (Log of Serum Triglycerides): The natural logarithm of the serum triglycerides level. Note: This is on a log scale (e.g., a value of 5.3 corresponds to approx 200 mg/dL).\
            s6 (Blood Sugar Level): Fasting blood glucose level (Glu).\
            target (Disease Progression): Indicates the progression of the disease one year after baseline. The variable is binarized based on the population mean: 0 indicates 'Lower Progression' (below average, more stable) and 1 indicates 'Higher Progression' (above average, rapid worsening).
            """.strip(),
}