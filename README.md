# Hafnia Training-as-a-Service: Breakout Session

Welcome to the Hafnia Training-as-a-Service (TaaS) breakout session at the Milestone Developer Summit.
In this session you will train, optimize and export a computer vision model on the
[Hafnia platform](https://hafnia.milestonesys.com/), first by clicking and then with code.

**What you need:** a laptop with a web browser, and a Google account for the notebook exercise (Google Colab).
You do not need to install anything locally.

**Contents**

- [Before the session: sign up and join the organization](#before-the-session-sign-up-and-join-the-organization)
- [Exercise 1: Launch your first training run](#exercise-1-launch-your-first-training-run)
- [Exercise 2: Customize and automate with the SDK and CLI](#exercise-2-customize-and-automate-with-the-sdk-and-cli)
- [Exercise 3: Export a model to OpenVINO with INT8 quantization](#exercise-3-export-a-model-to-openvino-with-int8-quantization)
- [Useful links](#useful-links)

---

## Exercise 1a: Sign up and join the organization
It takes a few minutes and lets us start training right away.

1. **Sign up and log in**
   Go to [hafnia.milestonesys.com](https://hafnia.milestonesys.com/), click **Register**  and create an account.
   Use the same email address you used to register for the breakout session, because that is where
   we sent your invitation. Then log in.

2. **Accept the invitation to the `BreakoutSessionDevSummit` organization**
   Open [Account settings → Invitations](https://hafnia.milestonesys.com/dashboard/account-settings/invitations)
   and accept the invitation from `BreakoutSessionDevSummit`.
   This organization gives you **free compute** for training during the breakout session.

3. **Check that you are in the organization**
   If you belong to more than one organization, make sure `BreakoutSessionDevSummit` is the active one
   before you create experiments. Otherwise your experiments will not use the free compute.

> **No invitation?**
> Check that you signed up with the email address you registered for the session with.
> If the invitation is still missing, send us your email address using
> [this form](LINK-TO-INVITATION-FORM) and we will invite you.

---

## Exercise 1b: Launch your first training run

**Goal:** Train an image classification model without writing any code.
**Time:** about 10 minutes.

You will train a ResNet-18 image classifier on the `mnist` dataset, using a public trainer provided by Hafnia.

1. Log in at [hafnia.milestonesys.com](https://hafnia.milestonesys.com/) and check that
   `BreakoutSessionDevSummit` is your active organization.
2. Go to [Training-aaS → Experiments](https://hafnia.milestonesys.com/dashboard/training-aas/experiments)
   and click **Create Experiment**.
3. **Dataset:** select `mnist`.
4. **Trainer package:** choose **Or select existing trainer package**, open the **Public Trainers** tab
   and select the **Image Classification Trainer (public)** trainer.
5. **Command Builder:** You are now navigated to the command builder. It shows the command that will be run for training, and lets you change arguments.
   Use the command builder to try changing arguments, for example the number of `epochs` to 5.
6. **Hardware:** select the `lite` configuration.
7. **Name:** Add a name to your experiment. Select something unique of your own choosing, for example `jane-doe-mnist`.
8. Click **Create Experiment**.
9. **Recommendation**: After launching the experiment, you will be navigated to the experiment page. All breakout session participants share this organization, so click the "My Experiments" chip to see only your experiments.
10. Click your experiment to open its page. You can see the training metrics appear in real time as the experiment runs.
**You are done when** your experiment appears in the experiment list, its status changes to running, and training metrics start to appear.

**Want more?** While it trains, try one of these:

- Create a second experiment that uses a **dataset recipe** instead of a dataset.
- Look at the other scripts in the trainer (for example `scripts/benchmark.py`) and the arguments the
  command builder offers for each one.

---

## Exercise 2: Customize and automate with the SDK and CLI

**Goal:** Use the Hafnia Python SDK and CLI to work with datasets and recipes, and to launch training from code.
**Time:** about 15 minutes.

👉 **Open the notebook in Google Colab:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/milestone-hafnia/taas-breakout-session-milestone-dev-summit/blob/main/notebooks/ex2-customize-and-automate.ipynb)

The link opens your own private copy of the notebook in Colab, running on a free Google machine. You need a
Google account. When you run the first cell, Colab warns that the notebook was not authored by Google:
click **Run anyway**. Your changes are not saved when you close the tab, unless you click
**File → Save a copy in Drive**.

Follow the instructions in the notebook. You only need to provide two things:

- **Your name**, which is used to name the recipes and experiments you create.
- **Your API key**, which you create on the Hafnia platform. We will show you where during the session.

> Keep your API key private. Do not share it or paste it anywhere other than the notebook.

---

## Exercise 3: Export a model to OpenVINO with INT8 quantization

**Goal:** Export a trained object detection model to OpenVINO and quantize it to INT8, so it runs
efficiently on Intel hardware.
**Time:** about 10 minutes.

The public **Object Detection Trainer** already contains pretrained RF-DETR models, so you do not need
to train or select a model first. You will run its export script on the built-in RF-DETR Nano model.

1. Go to [Training-aaS → Experiments](https://hafnia.milestonesys.com/dashboard/training-aas/experiments)
   and click **Create Experiment**.
2. **Dataset:** select "Vehicles Detection (Free)" (`midwest-vehicle-detection`). The export uses images from this dataset to calibrate
   the quantization.
3. **Trainer package:** choose **Or select existing trainer package**, open the **Public Trainers** tab
   and select the **Object Detection Trainer (Public)**.
4. **Command Builder:** press the `Change command schema` button and select the `scripts/export_openvino.py` script. Turn on **quantize**. Leave `model_path` at its default (`./pretrained_models/RFDETRNano.zip`). The command
   should look like this:

   ```bash
   python scripts/export_openvino.py --quantize
   ```

   Quantization needs a fixed input resolution, so do not leave `resolution` empty.
5. **Hardware:** select the `lite` configuration.
6. **Name:** include your name, for example `jane-doe-openvino-int8`.
7. Click **Create Experiment**.

**You are done when** the experiment has finished and its model files include the OpenVINO model
(`inference_model_int8.xml` and `inference_model_int8.bin`).

**What happened:** the script took the pretrained model bundled with the trainer, converted it to OpenVINO
format and applied post-training INT8 quantization. It used about 300 images from the hidden dataset to
calibrate the model, so the data never left the platform.

**Want more?** Run the same export on a model you trained yourself: select its checkpoint when you
create the experiment, and the script exports that checkpoint instead of the pretrained model.

---

## Useful links

- Hafnia platform: <https://hafnia.milestonesys.com/>
- Training-aaS documentation: <https://hafnia.readme.io/docs/training-as-a-service>
- Data library: <https://hafnia.milestonesys.com/dashboard/training-aas/data-library>
- Hafnia SDK and CLI: `uv add hafnia`
- Example trainers: 
    - [trainer-classification](https://github.com/milestone-hafnia/trainer-classification)
    - [trainer-object-detection](https://github.com/milestone-hafnia/trainer-object-detection)

