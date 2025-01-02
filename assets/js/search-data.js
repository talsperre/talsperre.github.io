// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "publications by categories in reversed chronological order. generated by jekyll-scholar.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-github-profile",
          title: "GitHub Profile",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "post-introducing-configurable-metaflow-by-netflix-technology-blog-dec-2024-netflix-techblog",
      
        title: 'Introducing Configurable Metaflow | by Netflix Technology Blog | Dec, 2024 | Netflix... <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
      
      description: "David J. Berg*, David Casler^, Romain Cledat*, Qian Huang*, Rui Lin*, Nissan Pow*, Nurcan Sonmez*, Shashank Srikanth*, Chaoying Wang*, Regina Wang*, Darin Yu**: Model Development Team, Machine…",
      section: "Posts",
      handler: () => {
        
          window.open("https://netflixtechblog.com/introducing-configurable-metaflow-d2fb8e9ba1c6", "_blank");
        
      },
    },{id: "post-supporting-diverse-ml-systems-netflix-tech-blog-netflix-techblog",
      
        title: 'Supporting Diverse ML Systems : Netflix Tech Blog | Netflix TechBlog <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
      
      description: "Machine Learning Platform at Netflix provides an entire ecosystem of integrations for practitioners so they can tackle a diverse set of business problems.",
      section: "Posts",
      handler: () => {
        
          window.open("https://netflixtechblog.com/supporting-diverse-ml-systems-at-netflix-2d2e6b6d205d", "_blank");
        
      },
    },{id: "news-our-work-infer-on-trajectory-forecasting-for-autonomous-vehicles-was-accepted-at-iros-2019",
          title: 'Our work INFER on trajectory forecasting for autonomous vehicles was accepted at IROS...',
          description: "",
          section: "News",},{id: "news-our-work-infer-on-trajectory-forecasting-for-autonomous-vehicles-was-accepted-at-iros-2019",
          title: 'Our work INFER on trajectory forecasting for autonomous vehicles was accepted at IROS...',
          description: "",
          section: "News",},{id: "news-our-work-on-characterizing-distracted-driving-in-social-media-has-been-accepted-at-aaai-icwsm-2020",
          title: 'Our work on characterizing distracted driving in social media has been accepted at...',
          description: "",
          section: "News",},{id: "news-our-work-on-e-challans-was-covered-in-the-times-of-india",
          title: 'Our work on E-Challans  was covered in The  Times of India.',
          description: "",
          section: "News",},{id: "news-graduated-with-a-bachelor-and-master-s-degree-in-computer-science-from-iiit-hyderabad-and-received-the-program-gold-medal-for-the-highest-gpa-in-the-graduating-class-of-2021",
          title: 'Graduated with a Bachelor and Master’s degree in Computer Science from IIIT-Hyderabad and...',
          description: "",
          section: "News",},{id: "news-i-was-awarded-the-jn-tata-endowment-scholarship-to-pursue-graduate-studies-in-the-us",
          title: 'I was awarded the JN Tata Endowment scholarship to pursue graduate studies in...',
          description: "",
          section: "News",},{id: "news-started-machine-learning-internship-at-the-ml-platform-team-at-netflix",
          title: 'Started Machine Learning Internship at the ML Platform team at Netflix.',
          description: "",
          section: "News",},{id: "news-started-machine-learning-internship-at-amazon-robotics",
          title: 'Started Machine Learning Internship at Amazon Robotics.',
          description: "",
          section: "News",},{id: "news-graduated-with-a-master-s-degree-in-computer-science-with-specialization-in-machine-learning-from-georgia-tech",
          title: 'Graduated with a Master’s degree in Computer Science with specialization in Machine Learning...',
          description: "",
          section: "News",},{id: "news-working-full-time-on-metaflow-as-a-machine-learning-engineer-in-the-ml-platform-team-at-netflix",
          title: 'Working full-time on Metaflow as a Machine Learning Engineer in the ML Platform...',
          description: "",
          section: "News",},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%73.%73%68%61%73%68%61%6E%6B%32%34%30%31@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/talsperre", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/shashanksrikanth", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=LjiaV8MAAAAJ", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/s_shawshank", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
