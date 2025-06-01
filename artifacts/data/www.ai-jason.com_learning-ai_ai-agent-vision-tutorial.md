[AI Agent](https://www.ai-jason.com/ai/ai-agent)

# AI agent + Vision = Incredible

# AI agent + Vision = Incredible

This video is sponsored by S Explain, the leading image-to-text platform. What would happen when an autonomous AI agent got GPT-4 Vision power? It would allow us to build an agent that can continuously iterate and improve web design, answer complex questions that are not possible to be answered today, and even power general-purpose robots where it can make plans and take actions based on the camera image it is taken. Many people have started experimenting with chat GPT Vision, but it wasn't clear what the boundaries are, what kind of image tasks it is doing well today, and what other tasks it is not doing well. How is prompting a multimodal model like GPT-4V different from other large language models that we have been using that only take text? Microsoft released a research paper that answers exactly those questions. They tested hundreds of different image tasks with GPT-4V to understand what it is actually good at and what it is not, and they also introduced some new prompting tactics. In this article, we will break this down for you so that you can understand what it is good at, what it is not, and how you can improve. In the end, we will show you a case study about how you can build an autonomous AI agent with vision ability today using autogen, stable diffusion, and Lava model to continuously self-improve AI-generated images.

AI agent + Vision = Incredible - YouTube

AI Jason

187K subscribers

[AI agent + Vision = Incredible](https://www.youtube.com/watch?v=JgVb8A6OJwM)

AI Jason

Search

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

Watch later

Share

Copy link

Watch on

0:00

/
•Live

•

[Watch on YouTube](https://www.youtube.com/watch?v=JgVb8A6OJwM "Watch on YouTube")

## The Power of Multimodal Models

If you are not familiar with multimodal models, here is a quick explanation. The large language models that we have been using today only take text input. They take a large amount of text data into vectors, so the next time someone prompts new text, it will start predicting the next words based on the vector space. A multimodal model, on the other hand, takes not only text data but also images, audio, and even video. It tries to tokenize all the different types of data and create a joint embedding so that it can understand these three types of data, even though they are in different formats. This unlocks some pretty crazy capabilities. For example, if you give the model an image or a park, as well as audio of a dog barking, it can return an image that is relevant to this scenario. It can also understand images of fridges and identify the items inside, even coming up with a menu for it. This definitely opens up lots of new use cases.

## Understanding Image Tasks

GPT-4V can handle loads of different image types. It can easily understand photographs, but it can also understand the text within the image very well, even if some of the text is distorted and hard to see. It can also understand formulas, tables, diagrams, and even floor plans. This is particularly exciting because there is a lot of data that couldn't be easily digitized and communicated to AI. Many knowledge bases are in PDF files, including diagrams and charts. GPT-4V can even summarize research papers based on a few pages of documents. It can extract data from images and understand them. For example, it can recognize people, landmarks, food brands, and logos, even if they are presented in a distorted way. It can also count the number of objects in an image and reason about relationships between objects. However, GPT-4V is not perfect and makes mistakes. For example, it may get the color of an object wrong or give incorrect answers when asked about charts or speed meters.

## Limitations of GPT-4V

Despite its impressive out-of-the-box performance, GPT-4V still has limitations. It may make mistakes in certain image tasks, and common prompting tactics may not always improve its performance. However, there are some prompting techniques that have been found to work well. Text instructions can provide more context and improve performance in certain tasks. Conditioned prompts, where the model is explicitly told the expectation, can also help. Few-shot prompts, where the model is given a few examples of how a task should be done, have shown significant improvements. Visual referring prompts, where the model understands visual annotations like arrows and circles, can also enhance performance.

## Prompting Techniques

Text instructions, conditioned prompts, few-shot prompts, and visual referring prompts are some of the prompting techniques that have been found to improve GPT-4V's performance. By using these techniques, you can fine-tune GPT-4V for specific image tasks and achieve better results. For example, manufacturers can build their own defect detection systems with sound training data, and medical professionals can train GPT-4V to assist in medical diagnoses. These techniques open up possibilities for training GPT-4V for niche image tasks.

## Case Study: Building an Autonomous AI Agent

Now let's explore a case study on how to build an autonomous AI agent with vision ability using autogen, stable diffusion, and Lava model. Although we don't have access to the GPT-4V API yet, we can use Lava as an alternative. Lava is a multimodal model based on Llama 2 that is good enough for building a proof of concept. In this case study, we will create an AI agent system that continuously improves the results of stable diffusion image generation. The system consists of one text-to-image engineer and one AI image analyzer. The text-to-image engineer uses autogen and stable diffusion to generate images based on prompts and improves the prompts based on feedback. The AI image analyzer uses Lava to review the generated images and provide feedback on how to improve the prompts. By combining these two agents, we can create an AI agent that continuously improves image generation results.

## Unlocking New Use Cases

GPT-4V unlocks a range of exciting use cases. One of them is building a real knowledge base for specific industries like architecture, engineering, and manufacturing. GPT-4V's ability to understand image and video data enables search across different types of data. Brands can search for videos where their logo has been presented, and users can search for specific information within images and videos. GPT-4V also opens up possibilities for building AI agents that can continuously improve image generation results. For example, GPT-4V can critique images generated from stable diffusion and provide feedback on how to improve them. This can create agents that continuously improve image generation results. Additionally, GPT-4V can be used for tasks like browser automation and robot control, enabling more sophisticated AI agents.

## Building a Real Knowledge Base

One of the exciting use cases of GPT-4V is building a real knowledge base for specific industries. Many knowledge bases are currently in PDF files, making it difficult for AI to access and understand the information within them. With GPT-4V's ability to understand images and extract data from them, it becomes possible to digitize and communicate this information to AI. This opens up opportunities for industries like architecture, engineering, and manufacturing to build knowledge bases that can be easily accessed and utilized by AI agents.

## Enabling Search Across Different Data Types

GPT-4V's multimodal capabilities enable search across different types of data. Traditionally, search engines have focused on text-based search. However, with GPT-4V, it becomes possible to search for information within images and videos. Brands can search for videos where their logo has been presented, allowing them to monitor brand visibility. Users can search for specific information within images, such as identifying objects or extracting text from images. This opens up new possibilities for information retrieval and search capabilities.

## Agent Ability

GPT-4V's ability to understand images and videos enables the creation of AI agents with vision capabilities. These agents can perform tasks like image generation, image analysis, and video transcription. For example, an AI agent can generate photo-realistic images based on prompts, analyze images for specific features or objects, and transcribe videos that don't have any script. This unlocks a new type of interaction where users can simply circle or point to something in an image or video, and the AI agent will be able to understand and provide assistance. This can be particularly useful for customer support or other applications where visual communication is important.

## Conclusion

GPT-4V, with its multimodal capabilities, opens up a world of possibilities for AI agents with vision abilities. It can understand and generate images, analyze images for specific features, and transcribe videos. It unlocks new use cases in various industries, enables search across different types of data, and allows for the creation of AI agents that continuously improve their performance. Whether it's building a real knowledge base, enabling sophisticated browser automation, or creating autonomous AI agents, GPT-4V's vision capabilities are truly incredible.

‍

🔗 Links

\- Follow me on twitter: [https://twitter.com/jasonzhou1993](https://twitter.com/jasonzhou1993)

\- Github: [https://github.com/JayZeeDesign/vision-agent-with-llava](https://github.com/JayZeeDesign/vision-agent-with-llava)

## Frequently Asked Questions

### 1\. Can GPT-4V understand distorted text within images?

Yes, GPT-4V has the ability to understand distorted text within images. It can accurately extract and interpret text, even if it is difficult to see or distorted.

### 2\. Can GPT-4V recognize specific objects or people in images?

Yes, GPT-4V can recognize specific objects, people, landmarks, food brands, and logos in images. It has the ability to understand and identify various elements within an image.

### 3\. How can GPT-4V be used to improve image generation results?

GPT-4V can be used to improve image generation results by continuously iterating and refining the prompts used for image generation. By providing specific feedback and using prompting techniques, the image generation process can be fine-tuned to achieve better results.

### 4\. What are some potential use cases for AI agents with vision abilities?

AI agents with vision abilities have a wide range of potential use cases. They can be used for tasks like image generation, image analysis, video transcription, browser automation, and robot control. They can also be used to build real knowledge bases for specific industries and enable search across different types of data.

### 5\. How can I start building my own AI agent with vision ability?

To start building your own AI agent with vision ability, you can use platforms like Autogen, Stable Diffusion, and Lava. These platforms provide the necessary tools and models to generate and analyze images. You can also experiment with different prompting techniques to improve the performance of your AI agent.

## Related articles

[Browse all articles](https://www.ai-jason.com/)