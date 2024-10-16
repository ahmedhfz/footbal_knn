import pandas as pd
import numpy as np

# Veriyi oku ve başlıkları al
data = pd.read_csv("test.csv")
data.fillna(0, inplace=True)
# Kaleci (GK) verilerini seç ve NaN değerleri ata
gk_data = data.loc[data["Positions"] == "GK"]

# Şut (Shooting) Özellikleri ve Ağırlıkları
shooting_features = ['Finishing', 'Long Shots', 'Free Kick Taking', 'Heading', 'Penalty Taking']
shooting_weights = {
    'Finishing': 0.40,
    'Long Shots': 0.30,
    'Free Kick Taking': 0.20,
    'Heading': 0.10,
    'Penalty Taking': 0.20
}

# Pas (Passing) Özellikleri ve Ağırlıkları
passing_features = ['Passing', 'Crossing', 'Corners', 'Long Throws']
passing_weights = {
    'Passing': 0.40,
    'Crossing': 0.30,
    'Corners': 0.15,
    'Long Throws': 0.15
}

# Hız (Speed) Özellikleri ve Ağırlıkları
speed_features = ['Acceleration', 'Pace', 'Agility', 'Dribbling']
speed_weights = {
    'Acceleration': 0.25,
    'Pace': 0.25,
    'Agility': 0.25,
    'Dribbling': 0.25
}

# Fizik (Physical) Özellikleri ve Ağırlıkları
physical_features = ['Height', 'Weight', 'Balance', 'Jumping Reach', 'Natural Fitness', 'Strength', 'Stamina']
physical_weights = {
    'Height': 0.10,
    'Weight': 0.10,
    'Balance': 0.15,
    'Jumping Reach': 0.15,
    'Natural Fitness': 0.15,
    'Strength': 0.15,
    'Stamina': 0.20
}

# Defans (Defensive) Özellikleri ve Ağırlıkları
defensive_features = ['Marking', 'Tackling']
defensive_weights = {
    'Marking': 0.50,
    'Tackling': 0.50
}

# Oyun Zekası (Game Intelligence) Özellikleri ve Ağırlıkları
game_intelligence_features = [
    'Ability', 'First Touch', 'Aggression', 'Anticipation', 'Bravery', 'Composure', 
    'Concentration', 'Decisions', 'Determination', 'Flair', 'Leadership', 
    'Off the Ball', 'Positioning', 'Teamwork', 'Vision', 'Work Rate', 'Technique'
]
game_intelligence_weights = {
    'Ability': 0.05,
    'First Touch': 0.05,
    'Aggression': 0.05,
    'Anticipation': 0.05,
    'Bravery': 0.05,
    'Composure': 0.05,
    'Concentration': 0.05,
    'Decisions': 0.05,
    'Determination': 0.05,
    'Flair': 0.05,
    'Leadership': 0.05,
    'Off the Ball': 0.05,
    'Positioning': 0.05,
    'Teamwork': 0.05,
    'Vision': 0.05,
    'Work Rate': 0.05,
    'Technique': 0.05
}

# Diğer Özellikler (Other) ve Ağırlıkları
other_features = ['Positions', 'Foot']
other_weights = {
    'Positions': 0.50,
    'Foot': 0.50
}

data['Height'] = data['Height'].str.replace(' CM', '').astype(float)
data['Weight'] = data['Weight'].str.replace(' KG', '').astype(float)
data["Positions"] = data["Positions"].str.split().str[0]

def get_weighted_rating(features, weights):
    # Özellik değerlerini float olarak al ve NaN olanları 0 ile değiştir
    feature_values = np.array([float(features.get(feature, 0)) for feature in weights])
    # Ağırlıkları da float olarak al
    weights_list = np.array([float(weights[feature]) for feature in weights])

    # Ağırlıklı ortalama hesapla
    calculated_rating = np.average(feature_values, weights=weights_list)

    return round(calculated_rating, 2)

def apply_rating(column_name , weight_list):
    data[column_name] = data.apply(
        lambda row: get_weighted_rating(
            {feature: row[feature] for feature in weight_list},
            weight_list
        ),
        axis=1
    )


apply_rating('Shooting Rating', shooting_weights)
apply_rating('Passing Rating', passing_weights)
apply_rating('Speed Rating', speed_weights)
apply_rating('Physical Rating', physical_weights)
apply_rating('Defensive Rating', defensive_weights)
apply_rating('Game Intelligence Rating', game_intelligence_weights)

columns_to_drop = shooting_features + passing_features + speed_features + defensive_features + game_intelligence_features + physical_features

# Sütunları kaldır
data = data.drop(columns=columns_to_drop)
data = data.drop(columns= "Foot")

# Veriyi tekrar ayır
gk_data = data.loc[data["Positions"] == "GK"]
no_gk_data = data.loc[data["Positions"] != "GK"]

gk_columns_to_drop = ["Defensive Rating", "Shooting Rating"]

# Sütunları GK veri setinden kaldır
gk_data = gk_data.drop(columns=gk_columns_to_drop)

# good_shooting_players = data[data['Shooting Rating'] > 70]
# filtered_shooting_players = good_shooting_players[["Shooting Rating"] + shooting_features]

# print(data[['Shooting Rating', 'Passing Rating', 'Speed Rating'
#             , 'Physical Rating', 'Defensive Rating', 'Game Intelligence Rating']].head())


# Pozisyonları ana gruplara ayıran bir sözlük
position_groups = {
    'GK': 'Kaleci',
    'DC': 'Defans',
    'DL': 'Defans',
    'DR': 'Defans',
    'WBL': 'Defans',
    'WBR': 'Defans',
    'DM': 'Orta Saha',
    'MC': 'Orta Saha',
    'ML': 'Orta Saha',
    'MR': 'Orta Saha',
    'AMC': 'Orta Saha',
    'AML': 'Orta Saha',
    'AMR': 'Orta Saha',
    'ST': 'Forvet'
}

# Mevcut "Position" kolonuna göre oyuncuları gruplandırma
data['Position Group'] = data['Positions'].map(position_groups)
# Yeni kolonun mevcut "Position" kolonu ile değiştirilmesi
data['Positions'] = data["Position Group"]

data = data.drop(columns="Position Group")

position_weightings = {
    'Kaleci': {'Shooting Rating': 0.0, 'Passing Rating': 0.1, 'Speed Rating': 0.1,
                'Physical Rating': 0.3, 'Defensive Rating': 0.4, 'Game Intelligence Rating': 0.1},
    'Defans': {'Shooting Rating': 0.05, 'Passing Rating': 0.15, 'Speed Rating': 0.1, 'Physical Rating': 0.3,
                'Defensive Rating': 0.3, 'Game Intelligence Rating': 0.1},
    'Orta Saha': {'Shooting Rating': 0.15, 'Passing Rating': 0.3, 'Speed Rating': 0.2, 'Physical Rating': 0.15,
                   'Defensive Rating': 0.1, 'Game Intelligence Rating': 0.1},
    'Forvet': {'Shooting Rating': 0.4, 'Passing Rating': 0.1, 'Speed Rating': 0.2, 'Physical Rating': 0.1,
                'Defensive Rating': 0.05, 'Game Intelligence Rating': 0.15}
}

def overall_calculator(row):

    weights = position_weightings.get(row["Positions"], {})

    return round(sum(row[feature]* weight for feature,weight in weights.items()),2)

data["Overall Rating"] = data.apply(overall_calculator, axis = 1)

data.to_csv("reytingli_test.csv")

print("DONE")














 
print("Done")



