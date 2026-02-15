# Car Damage Detection Model

Uses YOLOv8-seg to detect scratches and dents in cars.

*Input* : Images of damaged area by user
*Output* : Classification as to a scratch or dent, area of the dent or scratch and the damage location highlighted on the image

**Process**
1. Datasets used;
      * https://universe.roboflow.com/carpro/car-scratch-and-dent/dataset/1/download
      * https://cardd-ustc.github.io/  
2. Data annotation done using roboflow, it's API is used for training
3. Model training is done in Google Colab using T4 GPU
4. Model is saved in the Google Drive for using for prediction (not added to GitHub)
5. Model evaluation through analyzing metrics

*NOTES*
1. Install dependencies using: pip install -r requirements.txt