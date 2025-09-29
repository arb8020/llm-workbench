# Designing a Dashboard for Transparency and Control of Conversational AI - Methodology Quotes

## Synthetic Dataset Generation

> "We generated synthetic conversations using GPT-3.5 and LLaMa2Chat."

> "We used GPT-4 to annotate the generated data... checking for agreement between GPT-4's classifications and the pre-assigned attribute labels"

## Linear Probing Methodology

> "We trained linear logistic probes: pθ(X)=σ(⟨X,θ⟩)"

> "The linear probes were trained on the last token representation of a special chatbot message"

> "Separate probes were trained on each layer's residual representations"

## Causal Intervention Approach

> "We measured the causality of a probe by observing whether the model's response to a question changes accordingly as we intervene the relevant user attribute"

> "We created 30 questions with answers that might be influenced by it"

> "We used GPT-4 as a prompt-based classifier to compare the pairs of responses"

## User Interface Design

> "The TalkTuner UI consists of two main views... a standard chatbot interface and a dashboard"

> "We include a dashboard on the left to show how the chatbot is modeling the user"

> "Users can 'pin' the gender attribute with the arrow icons"

## User Study Design

> "We conducted a within-subject, scenario-based study where participants were asked to solve three tasks"

> "Participants were encouraged to think aloud as they completed tasks under three user-interface conditions"

## Implementation Details

> "The TalkTuner interface is a web application, implemented in Javascript with React"

> "We used the official checkpoint of LLaMa2Chat-13B released by Meta on HuggingFace"