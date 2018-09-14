import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM


#fetch data

data = fetch_movielens(min_rating = 4.0)


#print train and test

print(repr(data['train']))
print(repr(data['test']))


#create model

model = LightFM(loss = 'warp')

#train model

model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model,data, user_ids):

	#no. of users and movies in training data
	n_users,n_items = data['train'].shape

	#generate recom. for each user we input 
	for user_id in user_ids:

		#movie they already like
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		#movies we predict that they'll like
		scores= model.predict(user_id, np.arange(n_items))

		#rank them as best to least
		top_items = data['item_labels'][np.argsort(-scores)]

		#print out the results
		print("User %s"%user_id)
		print("		known_positives:")

		for x in known_positives[:3]:
			print("			%s"%x)

		print("		Recommended:")

		for x in top_items[:3]:
			print("			%s"%x)


sample_recommendation(model,data,[3,25,450])
