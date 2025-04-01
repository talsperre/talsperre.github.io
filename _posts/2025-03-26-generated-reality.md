---
layout: post
title: OpenAI Native Image Generation - Generative Reality, Generative UI,
  and more
date: 2025-02-16 09:00:00
description: Generative Reality - The next step in native image generation
tags: llms ar openai
categories: tech
featured: true
related_posts: false
images:
  compare: true
  slider: true
---

A short post on native image generation in LLMs, OpenAI's GPT-4o release and
Studio Ghibli trend, and applications of native image generation like
"Generative UI" and "Generative Reality".

## **OpenAI Native Image Generation**

OpenAI [announced](https://openai.com/index/introducing-4o-image-generation/)
native image generation in `GPT-4o` on March 25th, 2025. Immediately, after
the announcement, Machine Learning [TPOT](https://grok.com/share/bGVnYWN5_9177404c-2e12-4246-abc4-5b422ec48a9a)
(This Part of Twitter) was flooded with tweets about the image generation
capabilities of this model, particularly its ability to generate a version
of the provided image in the style of
[Studio Ghibli](https://en.wikipedia.org/wiki/Studio_Ghibli) animations,
initially popularized by Hayao Miyazaki and Isao Takahata.

<div class="container">
  <div class="row justify-content-center">
    <div class="col-md-6">
      {% twitter https://x.com/GrantSlatton/status/1904631016356274286 %}
    </div>
    <div class="col-md-6">
      {% twitter https://x.com/MDurbar/status/1904872441899339963 %}
    </div>
  </div>
</div>

The "offending" tweet (left one), which opened the floodgates, was from a user
named [`@GrantSlatton`](https://x.com/GrantSlatton/status/1904631016356274286).
Even though, generating ghiblified versions of images is fun, these native
image generation models are capable of much more than that. They can potentially
remove watermarks from images, add/remove subjects from images, help with
interior design and more. Taking it even further, a more advanced version of
such models can operate in real-time, altering the world we see and perceive
around us in real-time, leading to the term "Generative Reality". The later
sections of this post will explore this concept in more detail. What's
interesting is that despite all the hype about AI art and AI generated images,
the most popular use case right now seems to be modifying existing images of
friends, family, and pets in various styles. Imitation is the sincerest form of
flattery, and the fact that people are using these models to add filters on
top of their own photos, rather than creating entirely new images suggests
two things: True creativity is still exceptionally rare and very valuable,
and people still find immense joy in personal connections and memories.

*This newfound capability raises a natural question: what exactly is native
image generation, and how does it differ from earlier approaches? Let’s dive
into the technical details.*

### What is Native Image Generation?

Earlier image generation models like `DALL-E`, `CLIP`, and `Imagen` relied
on diffusion models, Vision Transformers, or Generative Adversarial Networks
(does anyone still remember GANs?). In contrast, newer models like `GPT-4o`,
`Grok 3`, and `gemini-2.0-flash-exp-image-generation` (yes, that’s really
the name 🤦‍️) are truly multimodal. These models can generate images and
audio in an autoregressive manner, much like they generate text. For
instance, `GPT-4o` produces images pixel by pixel, predicting the next pixel
one at a time. Multimodal models treat text, image pixels, and audio
waveforms as different tokens, training jointly across all three modalities.
The image and GIF below illustrate the inputs for a multimodal model and how
autoregressive text generation works in practice for unimodal models like
`GPT-2`.

<div class="row justify-content-center w-100">
    <div class="col-md-4">
        {% include figure.liquid path="/assets/img/post3/multimodal-llms.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <figcaption class="figure-caption text-center mt-2">Multimodal Large Language Model architecture</figcaption>
    </div>
    <div class="col-md-8">
        <figure class="figure">
            <img src="/assets/img/post3/gpt-2-autoregression-2.gif"
                 alt="Autoregressive generation"
                 class="img-fluid rounded z-depth-1"
                 onclick="this.classList.toggle('zoomed')" />
            <figcaption class="figure-caption text-center
mt-2">Autoregressive text generation process in GPT-2 <a
href="https://jalammar.github.io/illustrated-gpt2/">[*]
</a></figcaption>
        </figure>
    </div>
</div>

<br>

The main advantage of this approach is efficiency: we no longer need
specialized systems for different modalities, streamlining both training and
inference. Additionally, these models leverage cross-modal relationships,
enhancing their contextual understanding of scenes.

### How Does It Really Work?

Multimodal LLMs fall into two main categories:
- Models that process multiple input modalities (e.g., images, audio, text)
  but only generate text as output, such as
  [LLaMA 3.2](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/).
- Models that can also generate images or audio as output, like `GPT-4o` and
  `Grok 3`.


####  Deeper dive into multi-modal models

<div class="row justify-content-center w-100">
<div class="col-md-12">
        {% include figure.liquid
path="/assets/img/post3/multimodal-llm-input.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <figcaption class="figure-caption text-center
mt-2">Processing image using VIT for multi-modal LLMs <a href="https://sebastianraschka.com/blog/2024/understanding-multimodal-llms.html">[*]
</a></figcaption>
</div>
</div>

The mechanism of processing multimodal inputs is roughly similar in both
these types of models but the output generation is different. The image
below shows how a multi-modal LLM can process an image as well as text
together to generate a response. These multi-modal models typically process
an image into a smaller chunks of size `16 x 16` or `32 x 32` pixels, in a
left to right, top to bottom manner. These chunks are fed in a sequential
manner to another model, typically a Vision Transformer (ViT), which processes
these image chunks and generates a representation for each chunk. These
intermediate representations are then fed into a linear projection layer,
which resizes the image representations to the same dimensionality as the
input text embeddings, and also ensures that the generated image embeddings
are in the same "latent space" as the text embeddings. This alignment is
done by training the model, specifically the projection layers on a large
dataset of text-image pairs, after the base LLM has finished training.
Models from different research groups use varying approaches for training,
especially regarding which layers to freeze and which to update, but it common
to only update the linear projection layer and image encoder
during training. For instance, see this snippet from the LLama 3.2 blog post:

> To add image input support, we trained a set of adapter weights that
> integrate the pre-trained image encoder into the pre-trained language model.
> The adapter consists of a series of cross-attention layers that feed image
> encoder representations into the language model. We trained the adapter on
> text-image pairs to align the image representations with the language
> representations. During adapter training, we also updated the parameters of
> the image encoder, but intentionally did not update the language-model
> parameters. By doing that, we keep all the text-only capabilities intact,
> providing developers a drop-in replacement for Llama 3.1 models.

This [blog post](https://sebastianraschka.com/blog/2024/understanding-multimodal-llms.html)
by Sebastian Raschka provides a more technical deep dive into how multi-modal
models are trained, and the state-of-the-art models in this space as of late
2024. However, the blog post does not cover the generation/decoding
process for multimodal models like `GPT-4o` and `Grok 3`, which not only
process multimodal inputs, but are also capable of generating images in an
autoregressive manner. The concept of generating images
in an autoregressive manner is not new, and has been explored as far back as
the 2016s in papers like [PixelRNN](https://arxiv.org/abs/1601.06759).
PixelRNN simply used an RNN to predict the next pixel in an image, given the
previous pixels. However, there have been several improvements to such
models in recent years, with most of the improvements coming in the way that i)
images are encoded (CLIP/VQ-VAE) and, ii) modifying the process of
autoregressive generation itself from sequential visual token generation in a
raster-scan order, to more complex generation processes like
Visual AutoRegressive modeling (VAR), which is autoregressive generation of
images as coarse-to-fine “next-scale prediction” or “next-resolution
prediction”. Visual AutoRegressive modeling was introduced in the 2024
NeurIPS Best Paper award-winning paper ["Visual Autoregressive Modeling:
Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/pdf/2404.02905).

- The model first generates a low-resolution image starting from a $$1
  \times 1$$ token map, and then progressively increases the resolution by
  making the transformer based model to predict the higher resolution map.
- Each higher resolution map is conditioned on the previous lower resolution
  map, thus making the generation process autoregressive.
- Importantly, the authors show empirical validation of te scaling laws and
  zero-shot generalization potential of the VAR model, which is markedly
  similar to those of other LLMs.

<div class="row justify-content-center w-100">
<div class="col-md-12">
        {% include figure.liquid
path="/assets/img/post3/var-auto-regressive-models.png" class="img-fluid
rounded z-depth-1" zoomable=true %}
        <figcaption class="figure-caption text-center
mt-2">A Visual AutoRegressive model generates images in a coarse-to-fine
manner <a href="https://arxiv.org/pdf/2404.02905">[*]</a></figcaption>
</div>
</div>

<br>

The diagram above shows the process of VAR generation as described in the
paper. This is not the only way to generate images in an autoregressive
manner, and it isn't clear if `GPT-4o` or other native image generation
models like `Gemini 2.0 Pro Experimental` and `Grok 3` use similar
techniques. There have been some speculations that the GPT-4o model performs
autoregressive generation of images in a raster-scan order (left to right,
top to bottom), with generation happening at all scales simultaneously.
Another speculation is that there are two separate models, `GPT-4o`
generates tokens in the image latent space, and a separate diffusion based
decoder generates the actual image - see tweets below. Given the artifacts
being generated in the ChatGPT UI like top-down generation, blurry
intermediates, etc., I believe the latter speculation is more likely.

<div class="container">
  <div class="row justify-content-center">
    <div class="col-md-6">
      {% twitter https://x.com/jon_barron/status/1905143840572583974 %}
    </div>
    <div class="col-md-6">
      {% twitter https://x.com/sang_yun_lee/status/1905411685499691416 %}
    </div>
  </div>
</div>

### **Is any of this new?**

**The resounding answer is no.**

In fact, some of the style transfer results shown earlier are reminiscent of
the results from [DeepDream](https://en.wikipedia.org/wiki/DeepDream)
and [Neural Style Transfer](https://www.v7labs.com/blog/neural-style-transfer)
work by Gatys et al. in their paper
["Image Style Transfer Using Convolutional Neural Networks"](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
in 2016. The main caveat being that the results weren't as impressive, and they used
completely different architectures and training methods to achieve the results.
Several other models like Google's Gemini are multi-modal as
well and support similar image generation capabilities.
[Here's](https://x.com/OriolVinyalsML/status/1899853815056085062) the Gemini
announcement for instance:

<div class="container">
  <div class="row justify-content-center">
    <div class="row-md-4">
      {% twitter https://x.com/OriolVinyalsML/status/1899853815056085062 %}
    </div>
  </div>
</div>

However, none of them managed to capture the public imagination like this
release from OpenAI. The main reason for the relative lack of fanfare
regarding the Gemini announcement was two-fold:

- The Gemini releases were not product centric. The Gemini image generation
  model is only available as a preview model in Google AI Studio, and via
  Gemini API, and not in the main Gemini web interface. GPT-4o, on the other
  hand is available for users in the main ChatGPT website, immediately after
  the announcement. This is where network effects of over 400 million Weekly
  Active Users (WAUs) of ChatGPT come into play.
- The other harsh truth is that the image outputs of the Gemini model are
  simply not as good as the GPT-4o model. The results of both the Gemini
  model and the GPT-4o model on the same prompts are shown later
  [here](#putting-openais-image-generation-to-the-test).

### Putting OpenAI's image generation to the test

<div class="container">
  <div class="row justify-content-center">
    <div class="col-md-4">
      <div style="position: relative;">
        <img-comparison-slider hover="hover" value="70">
          {% include figure.liquid path="assets/img/post3/shashank-original.png"
          class="img-fluid rounded z-depth-1" slot="first" %}
          {% include figure.liquid path="assets/img/post3/shashank-ghibli-openai.png"
          class="img-fluid
          rounded z-depth-1" slot="second" %}
        </img-comparison-slider>
        <div class="caption-overlay">OpenAI GPT-4o Ghibli Style</div>
      </div>
    </div>

    <div class="col-md-4">
      <div style="position: relative;">
        <img-comparison-slider hover="hover" value="70">
          {% include figure.liquid path="assets/img/post3/shashank-original.png"
          class="img-fluid rounded z-depth-1" slot="first" %}
          {% include figure.liquid path="assets/img/post3/shashank-ghibli-grok.jpg"
          class="img-fluid
          rounded z-depth-1" slot="second" %}
        </img-comparison-slider>
        <div class="caption-overlay">Grok 3 Ghibli Style</div>
      </div>
    </div>

    <div class="col-md-4">
      <div style="position: relative;">
        <img-comparison-slider hover="hover" value="70">
          {% include figure.liquid path="assets/img/post3/shashank-original.png"
          class="img-fluid rounded z-depth-1" slot="first" %}
          {% include figure.liquid path="assets/img/post3/shashank-ghibli-gemini.jpeg"
          class="img-fluid
          rounded z-depth-1" slot="second" %}
        </img-comparison-slider>
        <div class="caption-overlay">Google Gemini Ghibli Style</div>
      </div>
    </div>
  </div>
</div>

<br>

**Are the results worth the hype?**

I compare the quality of the various multimodal models such as OpenAI's `GPT-4o`,
XAI's `Grok 3`, and Google's `Gemini 2.0 Pro Experimental` by testing them on the
same input image and prompt. Following up on the Studio Ghibli trend, the
prompt was "Create image - Convert this to studio ghibli style". The final
image generated by each model is shown here for comparison. We can see that
both `GPT-4o` and `Grok 3` generate pretty reasonable and visually appealing
results. The Gemini model, on the other hand, does not capture the intent of the
prompt as well as the other two models. When running further tests on images with
more people, `GPT-4o` edges out `Grok 3` in a bunch of cases, but both these models
are roughly on par with each other.

`GPT-4o` is also great at generating images in the style of other artists,
animation studios, and art movements. For instance, it is capable of
generating art in the form of Ukiyo-e, Art Deco, Pixar, Minecraft, Ufotable,
and more. It is also capable of arbitrarily changing the background or
context of an image, as shown in the comparison slider on the bottom right.
These models are only going to get better with time, and it does raise some
concerns about what is real and what is not. But matter of fact is that this
technology is here to stay, and there is no choice but to adapt to it.

<div class="container">
  <div class="row">
    <div class="col-md-5">
        <div class="col-12">
          <swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
            <swiper-slide>
              <div class="position-relative">
                {% include figure.liquid loading="eager" path="assets/img/post3/shashank-ufotable-openai.png" class="img-fluid rounded z-depth-1" %}
                <div class="caption-overlay">Ufotable Style Portrait</div>
              </div>
            </swiper-slide>
            <swiper-slide>
              <div class="position-relative">
                {% include figure.liquid loading="eager" path="assets/img/post3/shashank-starry-night-openai.png" class="img-fluid rounded z-depth-1" %}
                <div class="caption-overlay">Starry Night Style</div>
              </div>
            </swiper-slide>
            <swiper-slide>
              <div class="position-relative">
                {% include figure.liquid loading="eager" path="assets/img/post3/shashank-minecraft-openai.png" class="img-fluid rounded z-depth-1" %}
                <div class="caption-overlay">Minecraft Style</div>
              </div>
            </swiper-slide>
          </swiper-container>
        </div>
    </div>

    <div class="col-md-3">
      <swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" rewind="true" style="height: 100%;">
        <swiper-slide>
          <div class="position-relative">
            {% include figure.liquid loading="eager" path="assets/img/post3/shashank-manga-openai.png" class="img-fluid rounded z-depth-1" %}
            <div class="caption-overlay">Manga Style</div>
          </div>
        </swiper-slide>
        <swiper-slide>
          <div class="position-relative">
            {% include figure.liquid loading="eager" path="assets/img/post3/shashank-art-deco-openai.png" class="img-fluid rounded z-depth-1" %}
            <div class="caption-overlay">Art Deco Style</div>
          </div>
        </swiper-slide>
        <swiper-slide>
          <div class="position-relative">
            {% include figure.liquid loading="eager" path="assets/img/post3/shashank-ukiyo-e-openai.png" class="img-fluid rounded z-depth-1" %}
            <div class="caption-overlay">Ukiyo-e Style</div>
          </div>
        </swiper-slide>
        <swiper-slide>
          <div class="position-relative">
            {% include figure.liquid loading="eager" path="assets/img/post3/shashank-pixar-openai.png" class="img-fluid rounded z-depth-1" %}
            <div class="caption-overlay">Pixar Style</div>
          </div>
        </swiper-slide>
      </swiper-container>
    </div>

    <div class="col-md-4">
      <div style="position: relative;">
        <img-comparison-slider hover="hover" value="70">
          {% include figure.liquid path="assets/img/post3/crater-lake.png"
          class="img-fluid rounded z-depth-1" slot="first" %}
          {% include figure.liquid path="assets/img/post3/sequoia.png"
          class="img-fluid
          rounded z-depth-1" slot="second" %}
        </img-comparison-slider>
        <div class="caption-overlay">Background Replacement</div>
      </div>
    </div>

  </div>
</div>

<br>

<style>
  .position-relative {
    position: relative;
    width: 100%;
  }

  .caption-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: rgba(0, 0, 0, 0.7);
    color: white;
    padding: 8px 15px;
    text-align: center;
    border-bottom-left-radius: 4px;
    border-bottom-right-radius: 4px;
  }
</style>

#### Is it a fad or a big deal?

The Studio Ghibli trend went viral on social media, and it was akin to the
original ChatGPT moment, albeit on a much smaller scale. Just like any viral
trend on social media, this too will eventually fade away. However, unlike
other highly hyped trends in the Generative AI space, such as video
generation models like [Pika AI](https://pika.art/login), Autonomous agents
like [Devin AI](https://devin.ai/), [Manus](https://manus.im/), and GPT
stores like [Custom GPTs](https://openai.com/index/introducing-gpts/), this
trend is most likely more secular. According to Sam Altman, Native image
generation has already resulted in OpenAI gaining over 1 million users in an
[hour](https://x.com/sama/status/1906771292390666325) and [overloaded their
servers](https://x.com/sama/status/1905296867145154688) resulting in them adding
temporary rate limits.

### Real world use-cases of this technology

Right now, the biggest use case for these models seems to be simply
generating cute/fun images in different styles. Even this simple use case
will result in the release of new consumer-facing applications and tools.
However, this technology also opens up several other interesting possibilities,
especially in the virtual and augmented reality space, as well as in the realm
of user interfaces.

#### **Generative Reality**

As the ability of these models improve over time,
and the latency of generating these images improves, there is a future where
these models can be used to generate images in real-time, and even alter the
notion of reality around us. This is what I call "Generative Reality", which
is the topic of the blog post. Generative Reality is not exactly a concept that's
defined in the literature, and it's been discussed in various forms in the
past in science fiction shows like [Pantheon](https://www.imdb.com/title/tt11680642/)
(must watch for science fiction fans), books like
[Ready Player One](https://en.wikipedia.org/wiki/Ready_Player_One), and
[Snow Crash](https://en.wikipedia.org/wiki/Snow_Crash). Most of these
books/shows however focus on the concept of Virtual Reality (VR) and
"Generative Reality" is slightly different. It is similar to Augmented
Reality except that instead of overlaying information on top of the real
world, "Generative Reality" refers to the process of altering the way the
world looks while keeping it grounded in reality. An example of replacing
the "real world" with a Studio Ghibli style world using a headset is shown
below.

<div class="row justify-content-center w-100">
<div class="col-md-6">
        {% include figure.liquid
path="/assets/img/post3/generative-reality-ghibli.png" class="img-fluid
rounded z-depth-1" zoomable=true %}
        <figcaption class="figure-caption text-center
mt-2"> Generative Reality - Replacing the real world with a Studio Ghibli
style world
</figcaption>
</div>
<div class="col-md-6">
        {% include figure.liquid
path="/assets/img/post3/generative-reality-daylight-savings.png" class="img-fluid
rounded z-depth-1" zoomable=true %}
        <figcaption class="figure-caption text-center
mt-2"> Generative Reality - Replacing the real world with a Studio Ghibli
style world
</figcaption>
</div>
</div>
<br>

##### Dystopian Nightmare: A solution to switching the clocks

The debate around Universal standard time and Daylight Savings Time
resurfaces during the first week of November and March every year, when the
clocks are set back or forward. Proponents of Universal Time argue it aligns
with natural circadian rhythms, while Daylight Savings advocates favor
extended evening daylight for leisure. Here’s a tongue-in-cheek (and
dystopian) idea: instead of changing clocks, Generative Reality could adjust
how the world looks based on personal preference. Prefer Daylight Savings?
Your headset could render the world as if it’s an hour ahead.

#### **Generative UI**

A much more practical use case for native image generation is
["Generative UI"](https://uxdesign.cc/an-introduction-to-generative-uis-01dcf6bca808)
— using generative models to create adaptive, context-aware user interfaces.

Generative UI can be used to create personalized user interfaces that adapt
to the user's needs and preferences in real-time. For example, if a user is
booking a business trip via Airbnb, the generative UI can automatically adjust
the layout, to focus more on important information like commute time to the
downtown, or the speed of Wi-Fi at the hotel. On the other hand, family
vacation booking might look more like existing Airbnb interfaces, with
greater emphasis on family-friendly amenities, safety features, and cost
comparisons as shown below. Unlike "Generative Reality", this is not a pipe
dream, and is already supported to a certain extent by tools like Vercel's
[AI SDK](https://sdk.vercel.ai/docs/ai-sdk-ui/generative-user-interfaces).

<div class="container">
  <div class="row justify-content-center">
    <div class="col-md-10">
      <div style="position: relative;">
        <img-comparison-slider hover="hover" value="60">
          {% include figure.liquid path="assets/img/post3/airbnb-family.png"
          class="img-fluid rounded z-depth-1" slot="first" %}
          {% include figure.liquid path="assets/img/post3/airbnb-business.png"
          class="img-fluid
          rounded z-depth-1" slot="second" %}
        </img-comparison-slider>
        <div class="caption-overlay">Generative UI for two different
usecases: Family Travel & Business Travel.</div>
      </div>
    </div>
  </div>
</div>

<br>

### **What's Next?**

In terms of immediate industry headwinds, companies like Adobe, Figma, and
stock photo platforms like Shutterstock and Unsplash are likely to be affected
the most by the rise of native image generation models.

- **Adobe and Figma** are obviously affected the most, as their tools are
  heavily used for graphic design, UI/UX design, mockups, and image editing.
  Even if the current models cannot fully replace the need for professional
  design tools completely, they will only get better at instruction
  following over time, and will replace the need for these tools amongst the
  more casual users and hobbyists.

- The other companies that face significant negative headwinds are stock photo
  hosting and sharing platforms like **Shutterstock and Unsplash**. The
  impact on these companies is not as clear-cut, since other Image
  generation models like Midjourney and Stable Diffusion have been around
  for a while, and have not significantly disrupted this market yet.

- There's also potentially positive headwinds for **AR/VR** companies like
  Meta, Apple, Samsung, and more, since these models can help with creating newer
  User interfaces that are likely to make the AR/VR experience more
  immersive and personalized.

- **Social media companies** like Meta, Snapchat, and TikTok could also
  integrate these tools into their platforms to allow users to add more
  interactive filters to their stories and posts, increasing user engagement
  and retention.

As far as the core LLM technology is concerned, the logical next step would
be to train these multi-modal models on even larger datasets, and even more
modalities like video, and 3D models. Companies like [World Labs](https://worldlabs.ai/blog)
are already working on 3D world generation using LLMs / Foundation Models, and
future frontier models like `Gemini-3.0`, `Grok 4`, `GPT-6` and more will
likely have support for autoregressive video and 3D world generation as well.
The priors for this happening are quite high, due to the existence of
several large datasets of videos and 3D models available on the internet:

- **Video datasets**: All the content in YouTube, TikTok, movies, and more
- **3D datasets**: Data from varioues sources like Lidar scans from
  autonomous vehicles, 3D assets in game engines like Unreal Engine, Unity,
  and Blender, and spatial reconstruction data from NeRFs (Neural Radiance
  Fields) deployed in smartphone cameras and other devices.
