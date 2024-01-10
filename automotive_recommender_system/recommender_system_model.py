import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing
from torch.utils.data import DataLoader
from collections import defaultdict
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AutomotiveDataset:
    def __init__(self, users, product, ratings):
        self.users = users
        self.product = product
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        users = self.users[item] 
        product = self.product[item]
        ratings = self.ratings[item]
        
        return {
            "users": torch.tensor(users),
            "product": torch.tensor(product),
            "ratings": torch.tensor(ratings),
        }

class RecSysModel(nn.Module):
    def __init__(self, n_users, n_products):
        super().__init__()
        
        self.user_embed = nn.Embedding(n_users, 32)
        self.product_embed = nn.Embedding(n_products, 32)

        self.out = nn.Linear(64, 1)

    def forward(self, users, product, ratings=None):
        user_embeds = self.user_embed(users)
        product_embeds = self.product_embed(product)
        output = torch.cat([user_embeds, product_embeds], dim=1)
        
        output = self.out(output)
        
        return output   

df=pd.read_csv('Automotive.csv')       

lbl_user = preprocessing.LabelEncoder()
lbl_product = preprocessing.LabelEncoder()
df.reviewerID = lbl_user.fit_transform(df.reviewerID.values)
df.asin = lbl_product.fit_transform(df.asin.values)

#Split training and testing data

df_train, df_test = model_selection.train_test_split(
    df, test_size=0.2, random_state=42
)

#Create train and test dataset

train_dataset = AutomotiveDataset(
    users=df_train.reviewerID.values,
    product=df_train.asin.values,
    ratings=df_train.overall.values
)

test_dataset = AutomotiveDataset(
    users=df_test.reviewerID.values,
    product=df_test.asin.values,
    ratings=df_test.overall.values
)


# Create train and test dataloaders

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=64,
                          shuffle=True) 

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=64,
                          shuffle=True) 

dataiter = iter(train_loader)
dataloader_data = next(dataiter) 

# Create Model

model = RecSysModel(
    n_users=len(lbl_user.classes_),
    n_products=len(lbl_product.classes_),
).to(device)
optimizer = torch.optim.Adam(model.parameters())  
sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

loss_func = nn.MSELoss()




# Run the training model

epochs = 1
total_loss = 0
step_cnt = 0
plot_steps, print_steps = 5000, 5000
all_losses_list = [] 
gradient_accumulation_steps = 8

model.train() 
for epoch_i in range(epochs):
    for i, train_data in enumerate(train_loader):
        output = model(train_data["users"], 
                       train_data["product"])
        
        rating = train_data["ratings"].view(-1,1).to(torch.float32)

        loss = loss_func(output, rating)
        loss = loss / gradient_accumulation_steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item()    

        step_cnt = step_cnt + len(train_data["users"])
        
        if(step_cnt % plot_steps == 0):
            avg_loss = total_loss / (len(train_data["users"]) * plot_steps)
            # print(f"epoch {epoch_i} loss at step: {step_cnt} is {avg_loss}")
            all_losses_list.append(avg_loss)
            total_loss=0

    # Validate the model and adjust learning rate
    model.eval()
    with torch.no_grad():
        validation_loss = 0
        for j, test_data in enumerate(test_loader):
            val_output = model(test_data["users"], 
                               test_data["product"])
            val_rating = test_data["ratings"].view(-1, 1).to(torch.float32)
            val_loss = loss_func(val_output, val_rating)
            validation_loss += val_loss.item()
            
        
        # Adjust learning rate based on validation loss
        sch.step(validation_loss)

    model.train()    

# Plot the average loss during training

plt.figure()
plt.plot(all_losses_list)
plt.title("Average Loss vs Step Count")
plt.xlabel("Step Count")
plt.ylabel("Average Loss")
plt.draw() 
plt.show()    


###Calculating MAE####
model_output_list = []
target_rating_list = []

model.eval()

with torch.no_grad():
    for i, test_data in enumerate(test_loader): 
        model_output = model(test_data['users'], test_data["product"])
        
        model_output_list.append(model_output.sum().item() / len(test_data['users']) )

        target_rating = test_data["ratings"]
        
        target_rating_list.append(target_rating.sum().item() / len(test_data['users']))

mae = mean_absolute_error(target_rating_list, model_output_list)
print(f"MAE: {mae}")

#####Calculating RMSE#######

rms = mean_squared_error(target_rating_list, model_output_list, squared=False)
print(f"RMSE: {rms}")

# Create Ranking list with top 10 products for each user

def recommended_products(model, n_items=10):
    user_ids = df_test['reviewerID'].unique()
    recommendations = {}
    
    for user_id in user_ids:
        purchased_products = df_train.loc[df_train['reviewerID'] == user_id, 'asin'].unique()
        all_items = df['asin'].unique()
        not_purchased_products = np.setdiff1d(all_items, purchased_products)
        user_ids_tensor = torch.tensor([user_id]*len(not_purchased_products))
        product_ids_tensor = torch.tensor(not_purchased_products)
        predicted_ratings = model(user_ids_tensor.to(device), product_ids_tensor.to(device)).detach().cpu().numpy().flatten()
        recommended_product_ids = not_purchased_products[np.argsort(-predicted_ratings)][:n_items]
        recommended_product_labels = lbl_product.inverse_transform(recommended_product_ids)
        original_user_id = lbl_user.inverse_transform([user_id])[0]

        recommendations[original_user_id] = recommended_product_labels
    
    return recommendations

recommendations_list = recommended_products(model)
#Commenting out the print statement for the recommendation list
print(recommendations_list)    


#Function to evaluate the ranking list

def evaluate_ranking(model, data_loader):
    model.eval()
    true_ratings = []
    predictions = []
    total_users = 0
    num_conversions = 0
    with torch.no_grad():
        for i,data in enumerate(data_loader):
            user_ids = data["users"]
            product_ids = data["product"] 
            ratings = data["ratings"] 
            user_ids = user_ids.to(torch.long).to(device)
            product_ids = product_ids.to(torch.long).to(device)
            ratings = ratings.view(-1,1).to(torch.float32).to(device)
            output = model(user_ids,product_ids) 
            true_ratings.extend(ratings.cpu().numpy().flatten())
            predictions.extend(torch.floor(output).cpu().numpy().flatten())
            num_conversions += len(predictions)
            total_users += sum(predictions)
    precision = precision_score(true_ratings, predictions, average='weighted', zero_division = 0)
    recall = recall_score(true_ratings, predictions, average='weighted', zero_division = 0)
    fmeasure = f1_score(true_ratings, predictions, average='weighted', zero_division = 0)
    conversion_rate = (num_conversions / total_users) * 100
    return precision, recall, fmeasure, conversion_rate


precision, recall, fmeasure, conversion_rate = evaluate_ranking(model, test_loader)
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F-measure: {fmeasure:.4f}')
print("Conversion Rate: {:.2f}%".format(conversion_rate))


# Evaluating Precision@10 and Recall@10

user_rating_true = defaultdict(list)
with torch.no_grad():
    precisions = {}
    recalls = {}
    k=10
    threshold=3.5
    for i, test_data in enumerate(test_loader): 
        users = test_data['users']
        product = test_data['product']
        ratings = test_data['ratings']
        
        output = model(test_data['users'], test_data["product"])

        for i in range(len(users)):
            user_id = users[i].item()
            product_id = product[i].item() 
            pred_rating = output[i][0].item()
            true_rating = ratings[i].item()
            user_rating_true[user_id].append((pred_rating, true_rating))

    for user_id, user_ratings in user_rating_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k])
        precisions[user_id] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[user_id] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0  


# Precision and recall can then be averaged over all users
print(f"precision @ {k}: {sum(prec for prec in precisions.values()) / len(precisions)}")

print(f"recall @ {k} : {sum(rec for rec in recalls.values()) / len(recalls)}")                  