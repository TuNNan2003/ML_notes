#### The machine learning landscape

##### Definition of Machine Learning

**Machine learning** is the field of study that 

##### Supervised/Unsupervised learning

###### supervised learning

In supervised learning, the *training set you feed to the algorithm **includes the desired solutions**, called **labels***.

*classification* is a typical supervised learning task. 
And another typical task is to predict a *target* numeric value, given a set of ***features*** called **predictors**. This sort of task is called ***regression***.

> In machine learning an *attribute* is a data type while a *feature* has several meanings depending on the context.
> *feature* generally means an attribute plus its value.*(`mileage = 1500`)*

Note that some regression algorithms can be used for classification as well, and vice versa.

Some of the most important supervised algorithms:

- k-Nearest neighbors
- linear regression
- logistic regression
- support vector machines (SVMs)
- decision trees and random forests
- neural networks

###### unsupervised learning

In *unsupervised learning*, the training data is **unlabeled**. The system tries to learn without a teacher.

Some important unsupervised learning algorithms

- Clustering
  - K-means
  - DBSCAN
  - Hierarchical cluster analysis
- Anomaly detection and novelty detection
  - One-class SVM
  - Isolation Forest

- Visualization and dimensionality reduction
  - Principal component analysis (PCA)
  - kernel PCA
  - locally linear embedding (LLE)
  - t-Distributed Stochastic Neighbor embedding

- Association rule learning
  - Apriori
  - Eclat

*Visualization algorithms* are good examples of unsupervised learning algorithms: you feed them a lot of complex and unlabeled data, and they output 2D or 3D representation of your data that can easily be plotted. These algorithms try to preserve as much structure as they can so that you can understand how the data is organized and perhaps identify unsuspected patterns.

 A related task is ***dimensionality reduction***, in which the goal is **to simplify the data without losing too much information**.
One way to do this is to merge several correlated features into one, which is called ***feature extraction***.

> It is often a good idea to try to reduce the dimension of training data using a dimensionality reduction algorithm before we feed it to another Machine Learning algorithm.
> This will make it run much faster, the data will take up less disk and memory space, in some cases it may also perform much better.

Another important unsupervised task is ***anomaly detection***. The system is shown mostly normal instances during training, so it learns to recognize them when it sees a new instance, it can tell whether it looks like a normal one or whether it is likely an anomaly.

Another task is ***novelty detection***, it aims to detect new instances that look different from all instances in the training set.
This requires a "clean" training set, *devoid of any instance* that we would like the algorithm to detect.

One common unsupervised task is ***association rule learning***, in which the foal is to **dig into large amounts of data and discover interesting relations between attributes**.

###### Semi supervised learning

Since **labeling data is usually time-consuming and costly**, we will often have *plenty of unlabeled instances, and few labeled instances*. Some algorithms can deal with data that's partially labeled, called ***semi supervised learning***.

Some photo-hosting services are good examples of this. Once you upload all your family photos to the service, it automatically recognizes that the same person A shows up in photos 1,5,11, while another person B shows up in photos 2,5,7. This is the **unsupervised part** of the algorithm *(clustering)*. Now all the system needs is for you to tell it who these people are. Just add one label per person and it is able to name everyone in every photo.

Most semi supervised learning algorithms are **combinations of unsupervised and supervised algorithms**. For example, *deep belief networks* (DBNs) are based on unsupervised components called *restricted Boltzmann machines* (RBMs) stacked on top of one another. RBMs are trained sequentially in an *unsupervised manner* and then the whole system if fine-tuned using *supervised learning techniques*.

###### Reinforce learning

*Reinforcement Learning* is a very different beast. The learning system, called an ***agent*** in this context, can **observe the environment, select and perform actions, and  *get rewards* in return**. It must *then learn by itself* what is the best strategy, called a ***policy***, to get the most reward over time.

A **policy** defines *what action the agent should choose when it is in a given situation*.

Many robots implement *reinforcement learning algorithms* to learn how to walk. Deep mind's AlphaGo program is also a good example of reinforcement learning. It learned its winning policy by analyzing millions of games, and then playing many games against itself.

##### Batch and Online learning

Another criterion used to classify machine learning systems is whether or not the system can **learn incrementally from a stream of incoming data**.

###### Batch learning

In ***batch learning***, the system is **incapable of learning incrementally**. It **must be trained using all the available data**. This will generally take a lot of time and computing resources, so it is typically done *offline*. 

First the system is **trained**, and then it is **launched into production and runs without learning anymore**. It just applies what it has launched. This is called ***offline learning***.

If we want a **batch learning system to know about new data**, we need to *train a new version of the system from scratch* **on the full dataset** *(not just the new data, but also the old data)*, then stop the old system and replace it with the new one.

The whole process of training, evaluating, and launching a machine learning system can be **automated fairly easy**, even a batch system can adapt to change.

Train a new version of the system from scratch is simple and often works fine, but *training the full set of data can take many hours*. Also, training on the full set of data *requires a lot of computing resources*. Moreover, if the *amount of data is huge*, it may even be impossible to use a batch learning algorithm. If the system need to *adapt to rapidly changing data*, then we need a more reactive solution.

###### Online learning

In *online learning*, we train the system incrementally by **feeding it data instances sequentially**, either individually or in small groups called *mini-batches*. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives.

Online learning is great for **systems that receive data as a continuous flow** and **need to adapt to change rapidly or autonomously**. It is also a good option if you have limited computing resources: once an online learning system has learned about new data instances, it does not need them anymore,so we can discard them. This can save a huge amount of space.

Online learning algorithms can also be used to **train systems on huge datasets that** cannot fit in one machine's main memory. (called *out-of-core learning*). The algorithm **loads part of the data, runs a training step on that data, and repeats the process** until it has run on all of the data. 

> **Out-of-core learning** is usually done ***offline***, so *online learning* can be a confusing name.
>
> Think of *online-learning* as ***incremental learning***.

One important parameter of online learning systems is **how fast they should adapt to changing data**, this is called the ***learning rate***.
If you set a *high learning rate*, then our system will ***rapidly adapt to new data, but also tend to forget the old data quickly***. Conversely, if we set a low learning rate, the system will **have more *inertia***, that is, it will *learn more slowly*, but is will also be less sensitive to noise in the new data or to sequences of nonrepresentative data points. 
One challenge with online learning is that *if bad data is fed to the system, the system's performance will gradually decline*. If it's a live system, the clients will notice. Tp reduce this risk, we need to **monitor the system closely and promptly switch learning off and possibly revert to a previous working state** If we detect a drop in performance. We may also want to monitor the input data and react to abnormal data. *(using an anomaly detection algorithm)

##### Instance-based *vs* model-based learning

One more way to categorize machine learning systems is by *how they **generalize***. Most machine learning tasks are about **making predictions**, which means given a number of training examples. the system needs to be able to *make good predictions for examples it has never seen before*.

###### Instance-based learning

Possibly the most trivial form of learning is simply to *learn by heart*.

If we create a spam filter this way, the filter would only flag all emails that are identical to emails that have been flagged.
We hope the filter can *be programmed to flag emails that are very similar to known spam emails*, which need a ***measure of similarity*** between two emails.

***instance-based learning***: the system learns the examples by heart, then **generalizes to new cases** by using a **similarity measure** to compare them to the learned examples.

###### model-based learning

Another way to generalize from a set of examples it to ***build a model of these examples*** and then **use that model to make *predictions***. This is called ***model-based learning***. 

Before we use the model, we need to *define the **parameter values***.
 And to decide  which value will make the model perform best, we need to specify a ***performance measure***, we can either define a ***utility function (or fitness function)*** that measures how *good* the model is. Or we can define a ***cost function*** that measures how *bad* it is.

> The word "model" can refer to a **type of *model*** (e. g. Linear Regression ), to a *fully specified **model architecture***, or to the ***final trained model*** ready to be used for predictions.
>
> *model selection* consists in **choosing the type of model and fully specifying its architecture**.
>
> *training model* means **running a algorithm to find the model parameter** that will make it best fit the training data.

##### Summary

- studied the data
- selected the model
- trained the model on the training data
- applied the model to make predictions on new cases

#### Main challenges of machine learning

The two things that can go wrong are *"bad algorithm"* and *"bad data"*

##### Insufficient quantity of training data

Machine learning takes a lot of data for most algorithms to work properly. Even for simple problems we need thousands of examples, and for complex problems may need more.

> Microsoft researchers  have showed that **very different machine learning algorithms**, including fairly simple ones, **performed well on a *complex problem* of natural language disambiguation *once they were given enough data***. This idea was further popularized later.
>
> However, small- and medium-sized datasets are ***still very common***, and it's not always easy or cheap to get extra training data.

##### Nonrepresentative training data

In order to ***generalize well***, it is crucial that the **training data be representative of the new cases we want to generalize to**.
A unrepresentative training set will make a model unlikely to make *accurate predictions*.

So it is crucial to used a training set that is representative for the cases we want to generalize to.
If the sample is too small. we will have ***sampling noise***. Even very large samples can be nonrepresentative if the *sampling method is flawed*, which is called ***sampling bias***.

##### Poor-quality data

It is obvious that if the training data is *full of errors, outliers, and noise*, it will make it **harder** for the system to **detect the underlying patterns**, so the system is less likely to perform well. It is well worth the effect to **spend time cleaning up the training data**. And the truth is, *most data scientists spend a significant part of time just cleaning dataset*.

##### Irrelevant  features



























