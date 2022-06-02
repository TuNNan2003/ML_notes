#### End-to-end machine learning project

The main steps we will work through an example project from end to end.

- Look at the big picture
- get the data
- discover and visualize the data to gain insights
- prepare the data for machine learning algorithms
- select a model and train it
- fine-tune the model
- present the solution
- launch, monitor and maintain the system

##### Working with real data

It's best to experiment with real-world data. And we'll use the *California Housing Prices dataset*. 

##### Looking at the big picture

The fist task is to *use California census data to build a model of housing prices in the state*. The dataset includes some metrics such as the population, median income, and median housing price for each block group. We'll use these data to estimate the median housing price for a given district.

> Since we are a well-organized data scientist, the first thing we should do is ***pull out our own machine learning project checklist***. 

###### Frame the problem

The first thing we should know is ***what exactly the business objective is***.

**Building a model is probably not the end goal**. How do we expect to use and benefit from the model is more important. Knowing the objective is important for it will determine how we *frame the problem*, *which algorithms to select, which performance measure we'll use to evaluate the model, and how much effort we will spend tweaking it*.

For this model, the output will be fed to another machine learning system with many other signals, and the downstream system will determine whether is is worth investing in a given area.

> Pipelines
>
> **A sequence of data processing components** is called a ***data pipeline***.
>
> Pipelines are very common in machine learning systems for there is a lot of data to manipulate and many data transformations to apply.
>
> Components typically run *asynchronously*.
> Each component pulls in a large amount of data, process it and spits out the result in another data store. 
> Some time later, the next component in the pipeline pulls this data and spits out its own output.
>
> Each component is **fairly self-contained**: the ***interface between components*** is simply the **data store**. This makes the system simple to grasp, and different teams can focus on different components. 
> Moreover, if a component breaks down, the downstream components can often continue to run normally *(at least for a while)*by just using the *last output* from the broken component. This makes the architecture quit **robust**.

The next thing is to know ***what the current solution looks like***. The current situation will often give us a reference for performance, as well as insights on how to solve the problem.

Typically, this work is done by experts: a team gathers up-to-data information about a distinct, and when they cannot get the median housing price, they estimate it using complex rules. This way is always costly and time-consuming, and the estimates are not great.

Now we'll frame the problem:
is it supervised, unsupervised, or Reinforcement learning ? Is it a classification task, a regression task, or something else ?

This problem is 
**supervised learning task**, since *we are given **labeled** training examples*. 
**regression task**, for *we are asked to predict a value*. More specifically, this is a ***multiple regression problem***, since the system will **use multiple features to make a prediction**. 
***univariate regression*** problem, since we are only *trying to predict a single value for each district*. If we are trying to predict multiple value per district, it would be *multivariate regression* problem.
**batch learning** for there is no continuous flow of data coming into the system, no particular need to adjust to changing data rapidly, and the data is small enough to fit it in memory.

> If the data were huge, we could
> *split the work across multiple servers* (using the **MapReduce technique**)
> Or use an *online learning* technique

###### Select a performance measure

The next step is to ***select a performance measure***,

A typical performance measure for regression problems is the **Root Mean Square Error**. *(**RMSE**)* 
RMSE gives an idea of how much error the system typically makes in its prediction, with a higher weight for large errors.
$$
\mathcal{RMSE(X,h)}=\sqrt{\frac{1}{m}\sum^{m}_{i=1}(h(x^{i})-y^{i})^{2}}
$$

> Notation
>
> This formula introduces several common Machine learning notations that we use.
>
> - $$m$$ is ***the number of instances in the dataset*** you are measuring RMSE on.
> - $$x^{i}$$ is a ***vector of all the feature values of the $i$ th instance in the dataset***. (excluding the label)
> - $$y^i$$ is its ***label***, the *desired output value for that instance*.
> - $$X$$ is a ***matrix containing all the feature values of all instances in the dataset***.
>   There is *one row per instance and the $$i$$ th row is equal to the transpose of $$x^i$$, noted $${(X^i)}^T$$ 
> - $$h$$ is the ***system prediction function***, also called _***hypothesis***_. When a system is given an instance's feature vector $$x^i$$, it outputs a predicted value $$\widehat y^i = h(x^i)$$ 
> - $$RMSE(X, h)$$ is the ***cost function*** measured on the set of examples using the *hypothesis h*.
>
> |                                  |                       |                    |
> | -------------------------------- | --------------------- | ------------------ |
> | scalar values and function names | lowercase italic font | $$y^i$$ \ $x^i$    |
> | vectors                          | lowercase bold font   | $$\textbf{x}^{i}$$ |
> | matrices                         | uppercase bold font   | $$\textbf{X}$$     |

When there are *many outlier districts*, we may consider using the ***mean absolute error***. (**MAE**, also called *average absolute deviation*)
$$
\mathcal{MAE(X,h)=\frac{1}{m}\sum^{m}_{i=1}\mid h(x^i)-y^i\mid}
$$
Both RMSE and MAE are ways to measure the distance between the *vector of prediction* and the *vector of target values*.

Various distances measures, or *norms* are possible:

- **Computing the root of a sum of squares (RMSE)** corresponds to ***Euclidean norm***:
  This is the notion of distance we are familiar with, also called the $l_{2}$ *norm*, noted  $$\mid\mid\cdot\mid\mid_{2}$$ 

- **Computing the sum of absolutes (MAE)** corresponds to the $l_{1}$ *norm*, noted $$\mid\mid\cdot\mid\mid_{1}$$
  This is sometimes called the ***Manhattan norm*** for it measures the distance between two points in a city if you can only travel along orthogonal city blocks.

- The **$l_{k}\  norm$ ** of a vector $v$ containing $n$ elements is defined as
  $$
  \mid\mid v\mid\mid_{k}=(\mid v_{0}\mid^{k} + \mid v_{1}\mid^{k}+\dots+\mid v_n\mid^{k})^{\frac{1}{k}}
  $$
  $l_0$ gives **the number of nonzero elements in the vector**

  $l_{\infty}$ gives **the maximum absolute value in the vector**

- The higher the *norm index*, **the more it focus on large values and neglects small ones**.
  This is why ***RMSE is more sensitive to outliers than the MAE***. But when outliers are exponentially rare, the RMSE performs very well and is generally preferred.

###### Check the assumptions

It is good practice to **list and verify the assumptions that have been made so far**, this can help you *catch serious issues early on*.

#### Get the data

###### download the data

In typical environments our data would be available in a relational database and spread across multiple tables\documents\files.

In this project, we will just download a single compressed file `housing.tgz`, which contains a comma-separated values (CSV) file called `housing.csv` with all the data.
It is more preferable to create a small function to download that data.

```python
import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()

fetch_housing_data()

import pandas as pd

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
# this function returns a pandas DataFrame object containing all the data
```

We'll take a look at the top five rows using the `head()` method. And each row represents one district.
![](F:\Typora\Machine_Learning\Hands-On Machine Learning\C2_take_a_look_at_the_data.png)

The **`info()`** method is useful to get a quick description of the data, in particular the *total number of rows*, *each attribute's type*, and the *number of nonnull values*.
![](F:\Typora\Machine_Learning\Hands-On Machine Learning\housing_info.png)

Notice that the `total_bedrooms` attributes has only `20433` nonnull values, meaning `207` districts are missing this feature.
The `ocean_proximity`'s type is ***object***, so it could hold any kind of python object. We can find out **what categories exist and how many districts belong to each category** by using the ***`value_counts()`*** method.

```python
housing["ocean_proximity"].value_counts()
```

![](F:\Typora\Machine_Learning\Hands-On Machine Learning\value_counts().png)

The **`describe()`** method shows a summary of the numerical attributes
![](F:\Typora\Machine_Learning\Hands-On Machine Learning\describe().png)
