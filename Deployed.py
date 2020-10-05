import streamlit as st
from fastai.vision import load_learner, torch
import PIL.Image
from google_drive_downloader import GoogleDriveDownloader as gdd
gdd.download_file_from_google_drive(file_id='1-06LhL49Ai-Bb9u0zMLccR1gYxcpLJMK',
                                    dest_path='./CV_Project/models/wids_weights1.pth',
                                    unzip=False)

def main():
	st.title("Oil Palm Prediction")
	st.set_option('deprecation.showfileUploaderEncoding', False)
	img = st.file_uploader("Upload a file", type="jpg")
	img_t = Image.open(img)
	st.image(img_t, caption='Uploaded Image.')
	model = load_learner(r"wids_weights1.pth")
	pred_class = model.predict(img_t)[0] # get the predicted class
	pred_prob = round(torch.max(model.predict(img)[2]).item()*100) # get the max probability
	if str(pred_class) == 1:
	    st.success("This image has Oil Palm with the probability of " + str(pred_prob) + '%.')
	else:
	    st.success("This image does not have Oil Palm with the probability of " + str(pred_prob) + '%.')
	    
	st.subheader('Developed by: Riddhi Mehta')
if __name__=='__main__':
     main()
