---
Author: Kyle Lukaszek
---
# Setting up your environment!

It's important that we work with the exact same environments, so that we can reproduce our work on the ComputeCanada cluster.
## 1. Install Team Repository:

From your command line, visit the directory in which you want to do all of your work from.

For example, I have a `~/Classes/AI` directory that I do my work in.

So go to your directory of choice, and then clone the project into the folder.
```bash
cd ~/your/class/directory/

git clone https://github.com/klukaszek/MRI-RayTracer.git
```

This should add an `MRI-RayTracer/` directory to your class folder.

Now you should enter the directory to continue on.
```bash
cd MRI-RayTracer/
```

## Setting up YOUR Personal Branch.

We are working in personal branches to avoid stepping on anyone's toes and to prevent any file losses.

Once you are in the `MRI-RayTracer/` directory, you can simply change to your own branch with
```bash
git switch -c <your-branch-name>
```

Make sure the switch worked with
```bash
git status
```

Whenever you have work that you are happy with and want to save to share, you need to go to your terminal and run
```bash
git add path/to/file
git commit -am "Description of the work I've done!".
```

When you want to share it to GitHub so that we can run it on the server, you have to run
```bash
git push -u origin <your-branch-name>
```
## 2. Install UV:

From your terminal, install the UV Python environment manager.
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

`uv` should now be installed, and you should be able to run `uv --version`.

## 3. Installing Dependencies:

If you DO NOT have a CUDA capable machine. Install dependencies with:
```bash
uv sync --extra cpu
```

Otherwise, for a CUDA based install, do:
```bash
uv sync --extra cu128
```

## 4. Setting up IPYKernel + Jupyter:
Once the dependencies are installed, you'll also have to create an IPYKernel for running code notebooks using Jupyter or VSCode.

To install the Kernel, simply run the following command from within the `MRI-RayTracer` directory.
```bash
uv run python -m ipykernel install --user
```

With this complete, you should now be able to run code from Jupyter notebooks with no problems!

### VSCode

Make sure VSCode is up-to-date, and that you have the Jupyter + Python Extensions installed in VSCode. When trying to open a notebook in the `notebooks/` directory in VSCode, you *should* be prompted to install the extensions if none are present.

Once everything is installed, you should be able to click on the `Select Kernel` button in the top right of the notebook to select your Kernel from the `.venv/` directory where your IPYKernel is installed.

You should see something like the image attached below.

![[Pasted image 20251107171618.png]]

Select your `.venv` 
### Jupyter Lab

If you don't want to work with VSCode, you can work with Jupyter Lab in the browser.

To start Jupyter Lab:

```bash
uv run jupyter lab
```

This will open a Jupyter Lab instance in your browser, and print logging information to your console.

![[Pasted image 20251107172146.png]]

## 5. Download Data:

Since the UCSF-PGDM dataset is too big for most of our laptops. We can attempt to train our model on smaller datasets that share the same multi-modal information and segmentation format. This way, we can make use of UCSF-PGDM when training on the cluster.

To accomplish this, we will be using the smaller BraTS 2023+ Adult GLI data, and the MU-Glioma-Post datasets that are much lighter to keep locally.

### BraTS 2023+

You'll have to go to [BraTS 2023 Downloads](https://www.synapse.org/Synapse:syn51156910/wiki/627000]), and scroll to the bottom to access the data. The direct link to the data is [here](https://www.synapse.org/Synapse:syn64952532).

Once on the page, you want to click on the `BraTS-GLI` folder, and install the training data.
![[Pasted image 20251107175615.png]]![[Pasted image 20251107175633.png]]

To install the data, you'll have to make a Synapse account unfortunately, but once that's complete, make your way back to the downloads and download the training data.

## MU-Glioma-Post

You can download the MU-Glioma-Post dataset from the bottom of [here](https://www.cancerimagingarchive.net/collection/mu-glioma-post/)

You will need to follow the instructions and install the IBM Aspera Connect Extension, as well as the IBM Aspera Connect Client for your platform.

The instructions are pretty clear when you try to download the data.

![[Pasted image 20251107175923.png]]
![[Pasted image 20251107175950.png]]
### Data Organization

Once you have the data installed, you should move the zip files into a new folder in the `MRI-RayTracer/` project directory titled `data`.

Extract the zip archives into folders and rename them as shown below.

`BraTS-2023/`
`MU-Glioma-Post/`

![[Pasted image 20251107180306.png]]