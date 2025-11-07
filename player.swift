import pandas as pd

data = {
    "Points":[18,19,14,14,11,20,28,30,31,35,33,25,25,27,29,30,19,23,19,23],
    "Assists":[3,4,5,4,7,8,7,6,9,12,14,9,4,3,4,12,15,11,15,11],
    "Rebounds":[15,14,10,8,14,13,9,5,4,11,6,5,3,8,12,7,6,5,6,5]
}
pd.DataFrame(data).to_csv("players.csv", index=False)
print("✅ players.csv created successfully!")

