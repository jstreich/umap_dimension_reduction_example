################################################################################
# Example UMAP Projection, Dimension reduction plot and coordinates
# Authors: Jared Streich, ChatGPT for code rough draft, required debugging from
#   older libraries, 'n_componenets' edit, dot size, and column ranges read.
# Version 0.1.1
# Date: 2023-03-02
# email: streich.jared@gmail.com, ju0@ornl.gov
# UMAP is a machine learning based dimension reduction algorithm that projects
#   high dimensional data onto low dimensional space [manifold projection] while
#   optimizing the higher dimensional representation. UMAP is great for reducing
#   covariate space in a predictive model to fewer vectors that represent 
#   several descriptive dimensions.
################################################################################


################################################################################
############################## Import Libraries ################################
################################################################################

print("Loading Libraries...")
import pandas as pd
import umap
import matplotlib.pyplot as plt


################################################################################
############################### Read in Dataset ################################
################################################################################

##### Read in the data from the file
print("Reading in data...")
data = pd.read_csv('HS_145k-pixels_withNames_train_setDel.txt', sep='\t')

##### Extract the cluster IDs from the second column
print("Getting cluster IDs from Column 2...")
cl_ids = data.iloc[:, 1]

###### Get features
print("Setting remaining features as remaining columns 3:n...")
features = data.iloc[:, 2:].values


################################################################################
############################ Start Processing Data #############################
################################################################################

##### Perform UMAP projection with 2 dimensions
print("Calculating Dimension Reduction Projection...")
umap_out = umap.UMAP(n_components=2).fit_transform(features)

##### Create a scatter plot of the UMAP projection, colored by cluster ID
print("Creating Scatterplot...")
plt.scatter(umap_out[:, 0], umap_out[:, 1], c=cl_ids, s = 0.5)
plt.colorbar()

##### Save the plot to a PDF file
print("Saving Plot to File...")
plt.savefig('UMAP_plot_fnt0.5.pdf')

##### Save the UMAP data to a file
print("Writing Projection Manifold Coordinates to File...")
umap_vals = pd.DataFrame(umap_out, columns=['UMAP1', 'UMAP2'])
umap_vals.to_csv('UMAP_dim_output.txt', sep='\t', index=False)
