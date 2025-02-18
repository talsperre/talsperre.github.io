---
layout: post
title: LLMs, Game Theory, and Market Dynamics
date: 2025-02-16 09:00:00
description: Game theory of frontier LLMs
tags: llms ai tech
categories: tech
featured: true
related_posts: false
---

## **Grok 3 and other LLM announcements**

This week, Elon Musk announced that Grok 3 is going to be
released on February 17th, 2025. One reason to keenly watch the Grok 3
launch is that it's one of the first models that has been trained on orders of
magnitude more compute than previous gen GPT-4 class models. Grok 3 was pre-trained on the [Colossus
supercluster](https://www.tomshardware.com/tech-industry/artificial-intelligence/elon-musk-confirms-that-grok-3-is-coming-soon-pretraining-took-10x-more-compute-power-than-grok-2-on-100-000-nvidia-h100-gpus),
which contains over 100,000 NVIDIA H100 GPUs. The success of Grok 3 and
its performance on benchmarks will be a good indicator of whether the
scaling laws surrounding pre-training of LLMs still hold. The past two weeks have seen a flurry of activity in the LLM space. OpenAI just
announced that they will be launching GPT-4.5, codenamed Orion in a few weeks
time, and [GPT-5](https://arstechnica.com/ai/2025/02/sam-altman-lays-out-roadmap-for-openais-long-awaited-gpt-5-model/)
a few months after that. Additionally, there's
[rumors](https://techcrunch.com/2025/02/13/anthropics-next-major-ai-model-could-arrive-within-weeks/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAA7gdeeXoLhVq6eKujg65469N-Ep5I6Ul85jxeQneLRvQYJ30ivnje6AA0spHhCkSrfWmL3vn1iTwk-_If0xRPQyDL0lSMxB2cfbRhm8VPaSERmB1EB8qvF600GeXyRCyuAU27bxTmX-oY8IqsVbpcSxYfEZe09bacS8S3slhvnc)
that Anthropic is also planning on releasing a new model right around the same timeline.

<div class="row justify-content-center">
    <div class="col-md-4">
        <div class="row mb-4">
          {% twitter https://x.com/elonmusk/status/1890958798841389499 %}
        </div>
        <div class="row mb-3">
          {% twitter https://x.com/sama/status/1889755723078443244 %}
        </div>
    </div>
    <div class="col-md-6">
        {% twitter https://x.com/theinformation/status/1890484912748204381 %}
    </div>
</div>

This is a drastic departure from the past two years, where we saw a lot of
announcement of new models, but not many releases that represented a step
change in capabilities. [GPT-4](https://openai.com/index/gpt-4-research/)
was launched in May 2023, and the only other major update from OpenAI was
the launch of [GPT-4o](https://openai.com/index/hello-gpt-4o/) in May 2024,
and the launch of reasoning models like [o1](https://openai.com/index/introducing-openai-o1-preview/),
and [o3](https://openai.com/index/openai-o3-mini/). Unlike the jump in
capabilities from GPT-3.5 to GPT-4, the jump from GPT-4 to GPT-4o was minor.
The same applies to the other capability jumps like
[Claude Sonnet](https://www.anthropic.com/news/claude-3-family) to [Claude
Sonnet 3.5](https://www.anthropic.com/claude/sonnet), and from Gemini 1.5
Pro to Gemini 2.0 Pro. While reasoning models like o1 and o3 did show
significant improvements in benchmark performance, but are not directly comparable
to base LLMS like GPT-4 since they rely on inference time compute.

> Grok 3 was announced finally on 17th February 2025 and it is indeed a
> great model. The model is first on [LMArena leaderboard](https://x.com/lmarena_ai/status/1891706264800936307)
> with a score of over `1400` and its performance on [benchmarks](https://x.com/emollick/status/1891708599560253906)
> is comparable to OpenAI's `o3` models. The scale of improvements is not as
> drastic as the jump from GPT-3 to GPT-4, but the time it took xAI to catch up with
> other labs is impressive.

### Why the rush to release models now?

What's interesting is that all of these companies plan on releasing their
new LLMs within a few weeks of each other. Training these models often takes
several months, and the post-training, fine-tuning, and red-teaming adds a
few more months to this process. Given the long lead times, it's unlikely
that all of these companies just happened to finish training their models
at the same time. This in turn, suggests that most of these
companies have been sitting on their best models for a while, and are only
releasing them now due to competitive pressures.

But why would they do this? In a highly competitive market, it makes total
sense for companies to release their best models as soon as they are ready,
aim to capture as much market share as possible, and then iterate on the
next version. This has been the oft-repeated strategy in the tech industry
in the past, which was taken to its zenith by companies like Uber, and
Airbnb. The typical playbook during the ZIRP (Zero Interest Rate Policy) era
was to raise massive amounts of capital, utilize that capital to subsidize
services, and capture as much market share as possible. To hone in on the
point further:

- Uber first launched a beta/demo in San Francisco in 2010
- By 2012, Uber was serving rides across the US and in multiple cities
  across Europe.
- By 2013, Uber was already available in [74](https://www.theguardian.com/news/2022/jul/15/embrace-the-chaos-a-history-of-ubers-rapid-expansion-and-fall-from-favour)
  cities and was valued at $3.5  billion.

This paradigm of flooding the market with your best product using VC funding as
soon it is ready doesn't seem to apply to the LLM space. This is due to a unique
set of circumstances such as market forces like high interest rates,
competitive dynamics among companies due to the existence of knowledge
distillation, and the possibility of recursive self-improvement by acquiring
more user data. Let's explore each one of these in more detail.

## High Interest Rates

During the ZIRP era, raising capital was easy, and capital could be deployed
quickly to build out infrastructure, acquire users, and capture market share.
However, the higher interest rates of the past few years have made it
difficult to apply the same playbook. Additionally, the foundational
assumption of the earlier tech era companies was that the initial
infrastructure, R&D, and user acquisition costs would be high, but once a
user was acquired, the marginal cost of serving that user would be low. This
is what makes Google's AdWords, Facebook's marketplace / ad network, and
AWS's compute services so profitable. However, the marginal cost of serving
users in the LLM space is not low, and in fact, the marginal cost of serving
a user is significantly higher for models using inference time compute like
o1, o1 pro, and o3. The costs are so high that OpenAI has to charge over
$200 for their most premium [subscription](https://openai.com/chatgpt/pricing/)
which is the only way to access their latest o3 or o1 pro models.

The return to a world where marginal costs matter implies that companies
like OpenAI, Anthropic, Google, and others have to be more judicious in how
they deploy their capital, and make sure that they are able to recoup their
costs.

## Competitive market dynamics

There's two different types of competitive dynamics at play in the LLM space:

- User acquisition and training data acquisition
- Knowledge distillation

Both these dynamics are interrelated, and result in a game theoretic
situation where entities are incentivized to act in a certain way depending
on what they think other entities will do, and the other entities' actions.

### User Acquisition and Training Data Acquisition

LLMs are trained on vast amounts of data, and if the [scaling laws](https://arxiv.org/abs/2203.15556)
hold, then training larger models on more data will result in better models.
This implies that the entity that acquires the most users, and subsequently
the most data, will be able to train the best models, all other things being
equal. Thus, entities are incentivized to acquire as many users as possible,
which can be done by either releasing new state-of-the-art models that are
significantly better than the competition, or by offering the best models at
a lower price point.

### Knowledge Distillation

On the one hand, releasing new frontier models has positive network effects
for the entity releasing the model, its benefits are short-lived. This is
due to the existence of **Knowledge Distillation**, which allows other entities
to train smaller models that have similar performance to the original entity's
frontier model at a fraction of the cost. Distillation is very common in the
LLM space and is frequently used by companies to train smaller models that
can serve users at a lower price point. Some examples of this include
DeepSeek's V3 model, LLama 7B model, OpenAI's GPT-4o mini, and Google's
Gemini 2.0 Flash.

#### What is Knowledge Distillation?

**Knowledge Distillation** was first proposed in its modern form by Geoffrey
Hinton, Oriol Vinyals, and Jeff Dean in their 2015 paper
[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).
It's a fairly old concept, at least by modern ML standards, and at its core, is
just a process of training a smaller model to mimic the behavior of a larger model.
While a fairly simple concept, the implications of knowledge distillation being
possible are profound. Here's another way of looking at knowledge distillation:

- Say, you have a large model $$X$$ trained on a dataset $$D$$, with a certain
  performance metric $$Y$$.
- Now, you want to train a much smaller model $$X'$$ with smaller compute
  budget that has roughly similar performance to $$X$$.
- Instead of training $$X'$$ on $$D$$ directly, you instead train it on the
  predictions of $$X$$ on $$D$$.
- This surprisingly results in better performance than training $$X'$$ on $$D$$
  directly.

Here's a more authoritative excerpt from a 2021 paper "Does Knowledge
Distillation Really Work?" by [Stich et al.](https://arxiv.org/abs/2106.05237):

> Large, deep networks can learn representations that generalize well. While smaller, more efficient
networks lack the inductive biases to find these representations from training data alone, they may
have the capacity to represent these solutions. Influential work on knowledge
distillation argues that Bucila et al. “demonstrate convincingly that the knowledge acquired
by a large ensemble of models **[the teacher]** can be transferred to a single
> small model **[the student]**”.
Indeed this quote encapsulates the conventional narrative of knowledge distillation: a student model
learns a high-fidelity representation of a larger teacher, enabled by the teacher’s soft labels.

Several mechanisms for knowledge distillation have been proposed in
literature such as using a distillation loss to capture the difference
between the teacher and student predictions, using novel architectures and
distillation loss functions that minimize the difference between the feature
activations of the teacher and student, and more. Knowledge Distillation itself
can be applied at various stages of model training:

- In an offline setting, where the teacher model is trained first,
  and then the student model is trained on the teacher's predictions.
- In an online setting, where the teacher and student models are trained
  simultaneously in an end-to-end manner.
- Self-distillation, where the teacher and student models are the same,
  and the deeper layers of the model are used to train the shallower layers.

This [blog post](https://neptune.ai/blog/knowledge-distillation) goes into
greater detail on the various types of knowledge distillation, and specific
case studies.

#### Why does knowledge distillation matter?

As explained earlier, the entity releasing the best model first only has a
short window of time to reap the economic rewards from it, before others
distill from it and release smaller models that have similar performance.
The plot below shows how the compute budget required to train a new non-reasoning
frontier LLM has been increasing exponentially over the past few years. The compute budget for
Grok 3 pre-training is estimated to be around $$10\times$$ that of GPT-4, and
that of GPT-4 itself was rouhgly $$50\times$$ that of GPT-3. However, once a
new frontier LLM is released, it only takes roughly a year at max for other
companies to release much smaller models that have similar performance to the
previous frontier LLM.

<div class="l-page">
  <iframe src="{{ '/assets/d3/post2/llm_scaling.html' | relative_url }}"
    width="100%"
    height="720px"
    style="border: 1px dashed grey; overflow: hidden;"
  >
  </iframe>
</div>

For instance, DeepSeek's V3 model, a highly capable GPT-4 level model, was
trained on just a cluster of 2048 H800 GPUs, which is a fraction of the GPUs
used to train GPT-4. To achieve this drastic reduction in compute costs,
DeepSeek made several innovations such as:

- Leveraging Nvidia's assembly level PTX programming directly to bypass the
  limitations of the CUDA API, making it easier to achieve better GPU
  interconnect bandwidth.
- Using a novel multi-head latent attention layer that uses low rank
  compression of keys and values, to reduce memory footprint.
- Using DeepSeekMoE with support for hybrid routing, dynamic load balancing,
  and sequence wise balancing, to improve the efficiency of the model.

However, DeepSeek's V3 model also benefited from the fact that it was able to
distill from other frontier LLMs like OpenAI's GPT-4o, if OpenAI's / Microsoft's
[claims](https://www.theverge.com/news/601195/openai-evidence-deepseek-distillation-ai-data)
are to be believed. Thus, there is a game theory dynamic at play where entities
are incentivized to sit on their best models until the last possible moment, and
only release them when forced by market conditions.

## Recursive self-improvement

I personally don't ascribe to the view that LLMs will achieve recursive self-improvement
anytime soon, but it's worth mentioning as another factor that could be
driving the strategy of releasing frontier models amongst entities. The idea
of recursive self-improvement is that once an AI system reaches a certain
threshold of capability, it can then use that capability to improve itself
further, and so on. Think of it as a positive feedback loop where the AI
uses its current capabilities to improve itself exponentially, resulting in
an Artificial Superintelligence (ASI). For entities ascribing to this view,
there is no incentive to release new frontier models unless it is already an
ASI due to two reasons:

- Every small improvement in the model's current capabilities could be used
  to improve the model further, and releasing it to the public would allow
  other's to reverse engineer the improvements or distill from it. In order
  to truly reap the economic rewards of recursive self-improvement, it only
  makes sense to release the model once it is already an ASI.
- For entities concerned about "AI safety" or "AI alignment", releasing a
  model that is capable of recursive self-improvement could be catastrophic
  if the model is not aligned with human values. Thus, they would rather
  release models only after they have ensured that the model is aligned with
  human values, and is safe to use.

These values seem to align with strategies of companies like [Safe
Superintelligency (SSI)](https://ssi.inc/), and [Anthropic](https://www.anthropic.com/),

## **How is this related to game theory?**

The competing dynamics of user/data acquisition, knowledge
distillation, and recursive self-improvement can be modelled using game
theory. There are several competitors, which have partial information about
each other's capabilities, and are incentivized to act in a way that maximizes
their own utility, but not necessarily the utility of the group. The utility
for various entities is different, but at a high level, it can be thought of
as maximizing shareholder value. There are several other factors at play as
well, some of which can be confounding, such as regulatory constraints, ethical
considerations, and more. Thus, any attempt to model this using a simple
game theoretic framework would be an oversimplification, but it's still a
good approximation of the competitive dynamics at play - and can be used to
answer the original question of why release of frontier models are typically
clustered together.

### Modeling the LLM space as a two-stage coordination game

The forces of user/data acquisition and knowledge distillation are at odds
with each other, and can be modeled as a two-stage coordination game. Both
stages can be modelled as a coordination game, but the strategies and payoffs
are different for each stage.

#### What is a coordination game?

A [Coordination game](https://en.wikipedia.org/wiki/Coordination_game) is a
type of game where players benefit from making the
same or compatible choices. Assuming rationality, the key characteristic of
coordination games is that there are multiple equilibrium, and the main
challenge lies in selecting the equilibrium that is most beneficial for all
players. Selecting the optimal equilibrium is often difficult due to the
lack of communication between players, or having only partial information
about the other players' strategies. A
[Schelling point](https://en.wikipedia.org/wiki/Focal_point_(game_theory))
(or focal point) is a solution that players tend to choose in the absence of
communication because it appears naturally prominent or intuitive. Depending on
the payoffs, the Schelling point is often the optimal equilibrium for all players,
but in other cases, it might be worse for all entities involved.

The competitive dynamics in the LLM space can be modeled as a coordination
game because all entities can be assumed to be rational, and act in their own
self-interest, and only have partial information about the other's entities
strategies. In other words, in the absence of direct communication, the
entities will have to make decisions based on their own self-interest, and
also based on an understanding of what the other entities are likely to do.

#### Stag Hunt and Prisoner's Dilemma games

More concretely, the competitive dynamics can be modeled as a two-stage
coordination game, where the first stage is a "Stag Hunt" game, and the second
stage is a "Prisoner's Dilemma" game. The videos below provide a
good overview of the [Stag Hunt](https://en.wikipedia.org/wiki/Stag_hunt) and
[Prisoner's Dilemma](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma) games, and how the
payoffs are different for each game. The difference in payoffs between the
two games also results in a difference in strategies that players are
incentivized to take to maximize their utility.

<div class="row justify-content-center w-100">
    <div class="col-sm-6">
        <iframe
            width="100%"
            height="400"
            src="https://www.youtube.com/embed/oQ3KmcjwuKU"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen
            style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);">
        </iframe>
    </div>
    <div class="col-sm-6">
        <iframe
            width="100%"
            height="400"
            src="https://www.youtube.com/embed/jr2b0aZfOZQ"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen
            style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);">
        </iframe>
    </div>
</div>

> Videos were created by the `o1` model using [manim](https://www.manim.community/)
> and the prompts were created by yours truly.

In the context of frontier LLM releases, the first stage can be represented as a
**Stag Hunt** game, where entities are
incentivized to hold out on releasing their best models for as long as market
conditions allow. Sitting on the best model allows entities to capture as much
value out of their existing models and infrastructure as possible, and also
provides enough time to train the next frontier model. Releasing a new model
does provide a guaranteed short-term competitive advantage over other
entities, but in the long run, the benefits are not as clear, since it won't
take long for other entities to catch up via distillation or other means.
There are two equilibriums possible in this stage:

- **Equilibrium 1**: All entities hold out on releasing their best models,
  and only release them when forced by market conditions.
- **Equilibrium 2**: One entity releases their best model, and forces all
  other entities to release their best models as well. This implies that all
  entities release their best models at the same time.

<div class="row justify-content-center">
    <div class="col-sm-8">
        <iframe
            width="100%"
            height="500"
            src="{{ '/assets/d3/post2/stag_hunt_payoff_matrix.html' | relative_url }}"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen
            style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);">
        </iframe>
    </div>
</div>

The payoff matrix for the Stag Hunt game in the LLM space is shown above.
Due to the existence of partial communication among the various entities,
and partial trust in the other entities' strategies, the Schelling point in this
case is to hold out on releasing the best model, and it also happens to be the
optimal equilibrium for all entities involved. This is exactly what we are
witnessing in the last few months. However, as soon as one entity releases
or announces a plan to release a new frontier model, the market dynamics change,
and the second stage of the game begins.

The second stage of the game has different payoff structure, and can be
modelled as a **Prisoner's Dilemma** game. In this game, entities are forced
to release their best models to prevent other entities from capturing market share
at their expense. Given the nature of the payoffs, there is only one equilibrium in
this game:

- **Equilibrium 1**: All entities are forced to release their best models as
  soon as one entity announces a plan to release a new frontier model.

Unlike the Stag Hunt game, there is only one Nash equilibrium in the Prisoner's
Dilemma which is `(Defect, Defect)` (or `(Release, Release)` to be more
specific). Defection is the dominant strategy in this game, and since all
entities are aware of this, the Schelling point also becomes `(Defect, Defect)`.
However, unlike the Stag Hunt game, the Schelling point is not the pareto
optimal and is not the best outcome for all entities involved. Taken to its
extreme, this bodes poorly for the frontier LLM providers, as they will have
to constantly raise new capital to expand their infrastructure, and train
even larger models to stay ahead of the competition. Additionally, there are
other entities which aren't interested in developing frontier LLMs, but
nonetheless benefit from the existence of these models due to knowledge
distillation. In an ideal world, all entities would be better off if they
held out on releasing their best models, but the payoffs are such that the
dominant strategy is worse for all entities involved. The payoff matrix for
the Prisoner's Dilemma game in the LLM space is shown below.

<div class="row justify-content-center">
    <div class="col-sm-8">
        <iframe
            width="100%"
            height="500"
            src="{{ '/assets/d3/post2/prisoners_dilemma_payoff_matrix.html' | relative_url }}"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen
            style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);">
        </iframe>
    </div>
</div>

We have seen dynamic play out just a month ago, when OpenAI made
[o3 mini](https://openai.com/index/openai-o3-mini/) available for free to
all users, as soon as DeepSeek announced their R1 model. We are seeing the
same dynamic play out again, with OpenAI and Anthropic have announced their
intent to release new frontier models in the coming weeks, due to Grok 3's
imminent release. The visualization's below show how the payoffs and the
dominant strategies change between the two stages of the game.

## **Implications**

The above game theoretic model provides a good approximation of the
competitive landscape in the LLM space, but is not a perfect model. It
suffers from some drawbacks, chiefly:

- While most entities can be assumed to be rational and act in their own
  self-interest, i.e. maximize shareholder value, the means to achieve this
  may vary. For instance, companies like Meta and Amazon benefit from LLMs
  becoming commoditized since they are aggregators, and can thus release new
  models as soon as they are ready.
- If releasing frontier models is indeed the dominant strategy, then this
  might result in an arms race, where entities are incentivized to
  constantly train larger models to stay ahead of the competition. However,
  this means less time and resources are spent on important safety aspects
  like model alignment, and ethical considerations. **Note**: This is a
  highly controversial topic, and I am not suggesting that safety either be
  prioritized or deprioritized, but it's a tradeoff that entities will have
  to make.
- Releasing becoming the dominant strategy implies that frontier LLMs will
  become commoditized. This means that the economic rewards from releasing
  new models not only accrue to other competitors, but also to entities in
  other industries that can leverage these models to improve their own
  workflows. Thus, there is an argument to be made that policies should
  incentivize entities to release frontier models as soon as they are ready,
  for the "greater good".
