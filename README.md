# Sentiment analysis

Simple binary sentiment analysis via TensorFlow/Keras.

## Example

```
me$ python3 sentiment.py 

Training with 25000 entries (25000 labels
Token dictionary has 10003 entries

WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow_core/python/keras/initializers.py:119: calling RandomUniform.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 4)           40000     
_________________________________________________________________
global_average_pooling1d (Gl (None, 4)                 0         
_________________________________________________________________
dense (Dense)                (None, 4)                 20        
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 5         
=================================================================
Total params: 40,025
Trainable params: 40,025
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_impl.py:183: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 15000 samples, validate on 10000 samples
2019-09-18 10:26:41.309537: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fef0200a690 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2019-09-18 10:26:41.309561: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Epoch 1/40
15000/15000 [==============================] - 0s 19us/sample - loss: 0.6924 - acc: 0.5163 - val_loss: 0.6917 - val_acc: 0.5162
Epoch 2/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.6904 - acc: 0.5509 - val_loss: 0.6897 - val_acc: 0.5397
Epoch 3/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.6877 - acc: 0.5801 - val_loss: 0.6869 - val_acc: 0.6453
Epoch 4/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.6843 - acc: 0.6073 - val_loss: 0.6833 - val_acc: 0.6812
Epoch 5/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.6797 - acc: 0.6961 - val_loss: 0.6787 - val_acc: 0.7060
Epoch 6/40
15000/15000 [==============================] - 0s 10us/sample - loss: 0.6740 - acc: 0.7422 - val_loss: 0.6731 - val_acc: 0.7166
Epoch 7/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.6671 - acc: 0.7271 - val_loss: 0.6661 - val_acc: 0.7476
Epoch 8/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.6589 - acc: 0.7656 - val_loss: 0.6584 - val_acc: 0.7404
Epoch 9/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.6494 - acc: 0.7680 - val_loss: 0.6490 - val_acc: 0.7643
Epoch 10/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.6387 - acc: 0.7829 - val_loss: 0.6388 - val_acc: 0.7703
Epoch 11/40
15000/15000 [==============================] - 0s 10us/sample - loss: 0.6267 - acc: 0.7931 - val_loss: 0.6275 - val_acc: 0.7801
Epoch 12/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.6138 - acc: 0.7993 - val_loss: 0.6155 - val_acc: 0.7833
Epoch 13/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.6000 - acc: 0.8059 - val_loss: 0.6029 - val_acc: 0.7894
Epoch 14/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.5856 - acc: 0.8116 - val_loss: 0.5890 - val_acc: 0.7991
Epoch 15/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.5702 - acc: 0.8199 - val_loss: 0.5751 - val_acc: 0.8041
Epoch 16/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.5546 - acc: 0.8277 - val_loss: 0.5610 - val_acc: 0.8101
Epoch 17/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.5388 - acc: 0.8329 - val_loss: 0.5467 - val_acc: 0.8146
Epoch 18/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.5230 - acc: 0.8383 - val_loss: 0.5324 - val_acc: 0.8205
Epoch 19/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.5073 - acc: 0.8441 - val_loss: 0.5184 - val_acc: 0.8238
Epoch 20/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.4920 - acc: 0.8487 - val_loss: 0.5047 - val_acc: 0.8295
Epoch 21/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.4766 - acc: 0.8540 - val_loss: 0.4915 - val_acc: 0.8329
Epoch 22/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.4620 - acc: 0.8575 - val_loss: 0.4788 - val_acc: 0.8368
Epoch 23/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.4477 - acc: 0.8621 - val_loss: 0.4663 - val_acc: 0.8380
Epoch 24/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.4342 - acc: 0.8671 - val_loss: 0.4547 - val_acc: 0.8428
Epoch 25/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.4212 - acc: 0.8691 - val_loss: 0.4436 - val_acc: 0.8448
Epoch 26/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.4087 - acc: 0.8739 - val_loss: 0.4330 - val_acc: 0.8485
Epoch 27/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.3969 - acc: 0.8774 - val_loss: 0.4232 - val_acc: 0.8500
Epoch 28/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.3857 - acc: 0.8795 - val_loss: 0.4139 - val_acc: 0.8541
Epoch 29/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.3752 - acc: 0.8811 - val_loss: 0.4057 - val_acc: 0.8538
Epoch 30/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.3653 - acc: 0.8851 - val_loss: 0.3972 - val_acc: 0.8570
Epoch 31/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.3556 - acc: 0.8869 - val_loss: 0.3895 - val_acc: 0.8580
Epoch 32/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.3466 - acc: 0.8904 - val_loss: 0.3826 - val_acc: 0.8594
Epoch 33/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.3380 - acc: 0.8921 - val_loss: 0.3758 - val_acc: 0.8618
Epoch 34/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.3298 - acc: 0.8939 - val_loss: 0.3695 - val_acc: 0.8638
Epoch 35/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.3222 - acc: 0.8956 - val_loss: 0.3638 - val_acc: 0.8652
Epoch 36/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.3148 - acc: 0.8968 - val_loss: 0.3583 - val_acc: 0.8669
Epoch 37/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.3078 - acc: 0.8986 - val_loss: 0.3534 - val_acc: 0.8680
Epoch 38/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.3011 - acc: 0.9007 - val_loss: 0.3486 - val_acc: 0.8697
Epoch 39/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.2949 - acc: 0.9015 - val_loss: 0.3443 - val_acc: 0.8710
Epoch 40/40
15000/15000 [==============================] - 0s 11us/sample - loss: 0.2886 - acc: 0.9048 - val_loss: 0.3402 - val_acc: 0.8715
25000/25000 [==============================] - 0s 17us/sample - loss: 0.3507 - acc: 0.8623

Model metrics:
  loss : 0.3286221996021271
  acc : 0.8678399920463562

Sample predictions:

<START> this movie was so frustrating to watch the split screens don't allow you to get very involved in the emotions of the actors i was constantly going back and forth watching all these tiny images that i found myself with by the end this is basically a rip off of the talented mr <UNKNOWN> and <UNKNOWN> i was very let down with this film makers attempt to be cool i wish i had walked out like so many other people did
Sentiment: [0.3014337], Predicted label: 0, Actual label: 0

<START> this was a fantastically written screenplay when it comes to <UNKNOWN> things from another perspective the comedy was <UNKNOWN> and not overdone the acting was generally terrific and the plot line served a greater purpose of <UNKNOWN> <UNKNOWN> when we think about people solely based on their <UNKNOWN> appearance the plot twists as the brother sister character of amanda <UNKNOWN> tries to play soccer on the boys team finding instead a new love interest along the way <UNKNOWN> <UNKNOWN> is where the real lies and he does a fine job of acting <UNKNOWN> at first later coming to realize the most important thing in life is friendship not attitude
Sentiment: [0.6173827], Predicted label: 1, Actual label: 1

<START> try to look for another movie that is such a trip without having a story or plot and you'll be hard pressed head is a masterpiece of non linear non structure <UNKNOWN> and in less than 90 minutes it manages to be not only a time capsule of an era but also a full length experimental feature that defies time space and convention in a way that only underground films of the sixties could head is a reflection of those films no matter how one feels about the <UNKNOWN> this is a film every filmmaker should see because it cracks wide open the endless possibilities of film as an art medium had it not been for the film's <UNKNOWN> ad campaign and the fact that by the time it came out the <UNKNOWN> so the media that the public had become weary of them and every critic was ready to <UNKNOWN> on them this could have had a much greater impact studying how the film was edited is much more important and exciting than what's actually in the film and yet there are some great things in it great songs great cinematography etc should be seen after midnight for maximum effect because of it's overall <UNKNOWN> feel 1968 was a time of social <UNKNOWN> and a call for change thus the film's working title changes and head perfectly mirrors that time
Sentiment: [0.7548351], Predicted label: 1, Actual label: 1

<START> what on earth like watching an episode of <UNKNOWN> after drinking two <UNKNOWN> of <UNKNOWN> medicine nightmarish and making no sense at all i was waiting for the clever part where everything fits into place and saves the film maybe it was there and i just missed it or was lost on me br br my strongest suspicion is that it is a <UNKNOWN> <UNKNOWN> attempt to market a new drug thats about to hit the streets i wouldn't say don't watch it but i will say its pretty poor on every level like am in high <UNKNOWN> <UNKNOWN> unless you drink two <UNKNOWN> of <UNKNOWN> <UNKNOWN> then it's just dandy
Sentiment: [0.23391002], Predicted label: 0, Actual label: 0

<START> on the face of it this film looked like it might be really good it isn't br br the cast is pretty good but most of them seemed embarrassed by the whole thing the real disaster in this film is not the <UNKNOWN> but the script it attempts to include every cliché in the book all done incredibly poorly the ending is very abrupt but this is a <UNKNOWN> in disguise if you make it that far br br all three main male actors <UNKNOWN> <UNKNOWN> and <UNKNOWN> would surely agree that this is the low point in their careers i hope they got paid a lot of cash because none of their <UNKNOWN> come out in <UNKNOWN> br br the special effects are quite good but the same thing was done to much better effect in the day after tomorrow br br in short a pointless exercise don't waste your time
Sentiment: [0.05162698], Predicted label: 0, Actual label: 0

<START> focus is another great movie starring william h macy i first discovered macy in <UNKNOWN> and i've seen a few of his films and he hasn't yet <UNKNOWN> me macy is the <UNKNOWN> nice guy with something to hide in focus he plays the role of lawrence newman a loyal and hard working stiff who <UNKNOWN> his handicapped mother at home the scene is set after world war ii at the height of <UNKNOWN> newman is the head of human resources for a company which is basically anti <UNKNOWN> after he accidentally hires a woman of jewish descent he is asked to buy a pair of glasses to improve his failing <UNKNOWN> br br unbelievably the simple act of buying glasses has great <UNKNOWN> on his life and that of <UNKNOWN> hart his wife played by a great laura dern as the film <UNKNOWN> newman will begin to see a whole different world where being jewish is akin to being an animal br br the movie is disturbing in the way it shows that being racist was something fairly normal the chilling thought is that in some places it probably still is
Sentiment: [0.92869234], Predicted label: 1, Actual label: 1

the names <UNKNOWN> the mind but it's really just a coincidence all of said names were either just reaching the ends of their careers <UNKNOWN> <UNKNOWN> or beginning them everybody else br br only robert wise and <UNKNOWN> were in the middle of their careers br br for the record <UNKNOWN> was uncredited and learning his trade adam still had to <UNKNOWN> the <UNKNOWN> circle in the ceiling of sets a trademark he'd go on to put into all the early bonds baker had yet to star in and help produce the likes of <UNKNOWN> and robbery and go on to direct a <UNKNOWN> tv company called then die <UNKNOWN> young br br while harry andrews would go on to become one of <UNKNOWN> favourite character actors <UNKNOWN> scott daughter would never make the really big time but who can forget her in day of the <UNKNOWN> even though her bit was added later for padding and a happy ending or crack in the world br br sir cedric was theatre but knew how to <UNKNOWN> on film and <UNKNOWN> was <UNKNOWN> for br br but what were these stellar people doing in this camp old nonsense don't ask me the two main stars were no name italians helen had a <UNKNOWN> and paris was pretty while the brits were only there for support br br to <UNKNOWN> i think you can just mark this one up as a major <UNKNOWN> in <UNKNOWN> to be honest if i hadn't seen it i wouldn't believe it either
Sentiment: [0.10730705], Predicted label: 0, Actual label: 0

<START> a very good movie about anti <UNKNOWN> near the end of wwii the scene that really speaks <UNKNOWN> of the ignorance of these people is the meeting at the church when the priest is giving his speech against the international money and <UNKNOWN> it sounds amazingly like the speeches that <UNKNOWN> hitler used to force down his <UNKNOWN> throats yet none of the meeting <UNKNOWN> seem to make this comparison
Sentiment: [0.43621334], Predicted label: 0, Actual label: 1

is having an affair with a much younger woman eve bruce whom he also lies to in one very funny scene br br it's funny how the person whom we're looking for is the one who's always been there what could have been a <UNKNOWN> role for rick <UNKNOWN> who plays <UNKNOWN> sullivan next door neighbor turns into the man who not only sees the true beauty in fellow <UNKNOWN> stephanie but the one who saves toni at the start from killing herself not the stuff of comedy suicide then again this is not your average comedy and needless to say is ingrid <UNKNOWN> subtle poignant portrayal of a woman who's somehow missed her chances at love who's become <UNKNOWN> who due to a lie said to another she becomes the real person she was always meant to be i can't imagine anyone else in this quiet but deep role br br movies like these can be enjoyed at face value and seen as <UNKNOWN> fun a product of its times or be viewed for the deep symbolism that like its title it carries deep within it's a <UNKNOWN> film the same way <UNKNOWN> and <UNKNOWN> performance are equally <UNKNOWN> because in seeming so simple devoid of <UNKNOWN> and pose neither come out and <UNKNOWN> what they are about their acting becomes not really acting but playing real people <UNKNOWN> and all <UNKNOWN> flower is a story that never appears to take itself too seriously but reveals itself to be deep and very human after all
Sentiment: [0.635966], Predicted label: 1, Actual label: 1

<START> i agree with the user that this episode is awful <UNKNOWN> worst of the entire show now i'm not keen on many episodes of the later series but this one takes the <UNKNOWN> it was unfunny and as for the ending i'm sorry but it disgusted me more than any other episodes combined br br i mean the boys think they meant well but the ending was so <UNKNOWN> that they think the whale belongs on the moon and over the credits we see it has died could have saved the episode was if the <UNKNOWN> were able to confess for what they did br br there seem to be no <UNKNOWN> message okay south park may be guilty of preaching too much and its always nice to see an one such as make love not but this episode was just wrong avoid at all costs helen
Sentiment: [0.2303993], Predicted label: 0, Actual label: 0
```

## Notes:

Not much here at the moment! It's all extremely standard.
