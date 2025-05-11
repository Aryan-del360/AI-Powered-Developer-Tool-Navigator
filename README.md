
Project Title: AI-Powered Developer Tool Navigator
Tagline: Find the right developer tools instantly with AI-driven semantic search.


Section: Submission Categories

Select the following categories:
1. AI / Machine Learning
2. Cloud Computing
3. Databases
4. Developer Tools

Section: Demo Video Link

[PASTE YOUR PUBLIC DEMO VIDEO URL HERE]

Note:
     This YouTube video clearly shows your Streamlit app in action: entering a query, seeing results, using filters/sorting, viewing the chart, and reading the AI explanation.


Section: Code Repository Link

[(https://github.com/Aryan-del360/AI-Powered-Developer-Tool-Navigator)]

Note: 
      I have Uploaded "app.py" (or 'search_script_final.py'), 'README.md', and other necessary files (like the chart image) to a public repository on GitHub, GitLab.  
      

Section: Project Description

Problem: 

In the vast and ever-growing landscape of developer tools and technologies, finding the most relevant solutions based on a specific need or concept can be challenging with   
traditional keyword-based search. Developers need a more intuitive way to discover tools based on the "meaning" and context of their requirements.

Solution: 

Our project, the AI-Powered Developer Tool Navigator, addresses this by implementing a semantic search engine powered by cutting-edge AI and database technologies. It allows users  
to find developer tools not just by keywords, but by the underlying concepts and functionalities they describe.

How it Works (Technical Details):

1.  Query Embedding: 
                    When a user enters a search query in the Streamlit web interface, the text is sent to the "Google Cloud Vertex AI Text Embedding API ('text-embedding-004')" to  
                    generate a high-dimensional vector embedding that captures its semantic meaning.

2.  Vector Search: 
                  This query embedding is then used to perform a "Vector Search" on a dataset of developer tools stored in "MongoDB Atlas". We leverage MongoDB Atlas's native vector  
                  search capabilities to efficiently find documents (developer tools) whose embeddings are most similar to the query embedding.

3.  Result Processing 
    & Filtering: 
                     The initial search retrieves a set of relevant results. The Streamlit UI allows users to interactively filter these results based on topics and relevance  
                     score, and sort them to refine their view.
4.  Topic Analysis: 
                   The application analyzes the topics associated with the filtered results to generate a visual bar chart using `matplotlib`, providing a quick overview of the prevalent  
                   themes.
5.  AI Explanation: 
                   A summary of the top search results is fed into the "Google Cloud Vertex AI Gemini 2.0 Flash model". Gemini generates a concise, easy-to-understand explanation of the  
                   search results, highlighting their relevance from a developer's perspective.
6.  Interactive UI: 
                   A user-friendly web interface built with "Streamlit" provides the front-end for users to interact with the search engine, view results, apply filters, see the chart,            
                   and read the AI explanation. Features like Search History and Save to Favorites enhance the user experience.
7.  Secure Configuration: 
                         All sensitive credentials (MongoDB URI, GCP Project ID, HF Token) are managed securely using environment variables, following best practices.

Impact: 
This tool helps developers cut through the noise, quickly discover relevant tools based on semantic understanding, and gain insights into the key themes and functionalities of those tools through AI-generated explanations and visualizations. It streamlines the process of finding the right technology for their projects, particularly when building modern applications like SaaS products.

Highlighting Sponsor Technologies:
1. I extensively utilized "MongoDB Atlas Vector Search" as the core engine for performing efficient and scalable semantic similarity search on our dataset.
2. I integrated "Google Cloud Vertex AI" for two critical AI tasks: 
generating high-quality vector embeddings for semantic search and using the "Gemini 2.0 Flash model" to provide intelligent, context-aware explanations of the search results.


Section: Technologies Used

1. MongoDB Atlas (Vector Search)
2. Google Cloud Vertex AI (Text Embedding API, Gemini API)
3. Python
4. Streamlit (for the Web UI)
5. 'pymongo' (Python driver for MongoDB)
6. 'vertexai' (Python SDK for Google Cloud Vertex AI)
7. 'matplotlib' (for chart generation)
8. 'pandas' (for data handling in the UI)
9. 'requests' (if included for optional visual generation)
10. 'Pillow' (if included for optional visual generation)
11. Environment Variables for secure configuration


Section: Challenges

Developing this project involved overcoming several technical challenges:

1. Environment Variable Management: 
                                   Ensuring secure handling and correct loading of sensitive credentials ('MONGO_URI', 'GCP_PROJECT_ID', 'HF_API_TOKEN') across different operating system  
                                   terminals (Command Prompt, PowerShell) and ensuring they were accessible by the Python script and Streamlit application. This required careful syntax and  
                                   debugging.
2. API Integration: 
                   Successfully integrating the distinct APIs of MongoDB Atlas (via 'pymongo') and Google Cloud Vertex AI (for embeddings and generative models) into a single, cohesive  
                   application flow.
3. Debugging Runtime Errors: 
                            Identifying and resolving specific Python errors ('SyntaxError', 'NameError', 'AttributeError') that arose during the development and integration of different  
                            components and libraries within the Streamlit environment, including adapting to changes in Streamlit's API ('st.experimental_rerun' to 'st.rerun').
4. (Optional, if applicable): 
                             Handling potential rate limits or authentication complexities when experimenting with external APIs for visual generation (e.g., Hugging Face).


Section: What's Next / Future Enhancements

Given more time, we would like to enhance the project by:
1. Deployment: Deploying the Streamlit application to a publicly accessible platform like Streamlit Cloud or Google Cloud Run to make it easily available to users.
2. Data Ingestion Pipeline:  Building a more robust data ingestion process, potentially automating the fetching, embedding, and loading of new developer tool data into MongoDB Atlas.
3. Advanced AI Insights:     Implementing more sophisticated AI analysis, such as extracting structured key features from tool descriptions or suggesting related search queries using 
                             generative AI.
4. Expanded Dataset:         Incorporating a larger and more diverse dataset of developer tools.
5. User Features:            Enhancing features like saving favorite tools (persisting them beyond the session) and expanding search history capabilities.
6. Monitoring & Logging:     Integrating with Google Cloud Logging and Monitoring to track application performance and errors in a production environment.
7. Improved UI/UX:           Further refining the visual design and adding features like a dark/light mode toggle.


Other Sections:

Screenshots:         Include screenshots of Streamlit app's main page, search results, sidebar filters, and the generated chart.
Additional Files:    Include the "app.py" (or 'search_script_final.py') and 'README.md' in GitHub and GITLAB code repository. It has generated chart image.
