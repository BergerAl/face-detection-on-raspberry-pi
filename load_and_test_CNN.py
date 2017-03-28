from keras.models import model_from_json
from keras.preprocessing.image import array_to_img, img_to_array, load_img

#load model
json_file = open('models/basic_cnn_30_epochs_data.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

#load weights
model.load_weights("models/basic_cnn_30_epochs_data.h5")


#Test image
test_img = load_img('data/validation/Emma_Watson/pic_306.jpg', target_size=(150,150))
image_as_array = img_to_array(test_img)
image_as_array = image_as_array.reshape((1,) + image_as_array.shape)
prediction = model.predict(image_as_array)              # for vector output
#prediction = model.predict_classes(image_as_array)      # for classes output
print prediction