# Expression Analyzer

A web app that finds faces in an uploaded image and estimates visible facial
expressions across seven categories. Each detected face gets a probability
breakdown and an annotated image.

Estimates describe visible expressions, not a person's internal emotional state.
Do not use them for high-stakes decisions.

## Deploy to Render

The repository includes a Render Blueprint (`render.yaml`) configured for the
free web-service tier. To deploy it:

1. Push this project to a GitHub repository that you control. Keep the model and
   label files; `.gitignore` excludes local caches and editor settings.
2. In Render, choose **New → Blueprint**, connect your GitHub repository, and
   apply its `render.yaml` configuration.
3. Wait for the build and deploy to finish, then open the Render URL and check
   its `/health` endpoint.

The free tier has 512 MB RAM. TensorFlow may exceed this limit while starting;
if the service fails or restarts, select a Render plan with more memory. Free
services may also sleep when idle and take longer to respond on the first
request.

The model and labels are required by the app. The original FER-2013 dataset and
training artifacts are not needed for inference. Saved training outputs, if
available, are kept separately from this deployment folder.

## Use

1. Open **Analyze a Photo** from the home page.
2. Select, drag and drop, or paste a JPG, JPEG, or PNG image (maximum 10 MB).
3. Select **Analyze Photo** to see detected faces and expression estimates.

The JSON inference endpoints are `POST /predict` and `POST /api/predict`. Send
the image as multipart form data with the field name `image`.

