# Step-by-Step Process
SAM stands for Segment Anything. This is a [foundation model by Meta](https://segment-anything.com/). I have been using it for my medical image segmentation, and it has been extremely powerful for generating initial masks, on which my medical professional collaborators can add/remove wrong annotations. It reduces their work by 10x, I believe. However, SAM2 is not available offline in our own system - the live app system is extremely powerful because the prompts can be adjusted based on the live outputs that it gives. Now, you can do it in your own system. 

## Step 1: Install SAM2.
We are using the image predictor capability of SAM2. Follow the instructions [here](https://github.com/facebookresearch/sam2) clearly.

## Step 2: Add the code.
Create a folder named "app" inside the "sam2" folder, and add the code file shared in that folder.

## Step 3: Run the code.
Now, run the code and enjoy your in-house SAM code!

[Check out the live SAM2 application on my system here.
](https://www.linkedin.com/posts/srijit-mukherjee_code-%F0%9D%90%92%F0%9D%90%AE%F0%9D%90%A9%F0%9D%90%9E%F0%9D%90%AB%F0%9D%90%9F%F0%9D%90%9A%F0%9D%90%AC%F0%9D%90%AD-%F0%9D%90%92%F0%9D%90%9E%F0%9D%90%A0%F0%9D%90%A6%F0%9D%90%9E%F0%9D%90%A7%F0%9D%90%AD-activity-7388539178927493120-Iint?utm_source=share&utm_medium=member_desktop&rcm=ACoAAB2QfP8BepQmyPYA2Ly4YR-iNUAam41Nk2M)
--- 

## Clever engineering for precise results
Meta employs some clever engineering in its online version, which is only briefly mentioned in the image predictor IPython notebook file. Let me tell you the secret. "Predict with `SAM2ImagePredictor.predict`. The model returns masks, quality predictions for those masks, and low-resolution mask logits that can be passed to the next iteration of prediction." In this work, I use the same method with the prompts like Positive, Negative Points, along with the Bounding Box. This has been my dream project from the time I started using SAM. Very happy today.
