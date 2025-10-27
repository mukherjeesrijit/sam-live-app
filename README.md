# Step-by-Step Process
SAM stands for Segment Anything. This is a [foundation model by Meta](https://segment-anything.com/). I have been using it for my medical image segmentation, and it has been extremely powerful for generating initial masks, on which my medical professional collaborators can add/remove wrong annotations. It reduces their work by 10x, I believe. However, SAM2 is not available offline in our own system - the live app system is extremely powerful because the prompts can be adjusted based on the live outputs that it gives. Now, you can do it in your own system. 

## Step 1: Install SAM2.
We are using the image predictor capability of SAM2. Follow the instructions [here](https://github.com/facebookresearch/sam2) clearly.

## Step 2: Add the code.
Create a folder named "app" inside the "sam2" folder, and add the code file shared in that folder.

## Step 3: Run the code.
Now, run the code and enjoy your in-house SAM code!

[Check out the live SAM2 application on my system here.
](https://youtu.be/GunCUhhBzaE?si=veu3fK24cZ1B1ppN)
--- 

## Clever engineering for precise results
Meta employs some clever engineering in its online version, which is only briefly mentioned in the image predictor IPython notebook file. Let me tell you the secret. "Predict with `SAM2ImagePredictor.predict`. The model returns masks, quality predictions for those masks, and low-resolution mask logits that can be passed to the next iteration of prediction." In this work, I use the same method with the prompts like Positive, Negative Points, along with the Bounding Box. This has been my dream project from the time I started using SAM. Very happy today.
