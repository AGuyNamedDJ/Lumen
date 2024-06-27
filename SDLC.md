# Software Development Life Cycle (SDLC) Plan

This document outlines the high-level strategy and plan for developing the backend of the Lumen quantitative trading platform.

## Table of Contents <a name="table-of-contents"></a>

1. [Inception](#inception)
2. [Design](#design)
3. [Implementation](#implementation)
4. [Testing](#testing)
5. [Deployment & Maintenance](#d-m)
   - [Deployment](#deployment)
   - [Maintenance](#maintenance)

---

## Inception Overview <a name="inception"></a>

1. **Identified the key stakeholders**:
   - The main stakeholders of this project are myself, Dalron J. Robertson, and other traders I share it with. The system is built for my personal use and for enhancing the trading strategies of others.
2. **Gathered requirements**:
   - Through my extensive experience and analysis of existing systems, I determined the essential features for the backend system. These included managing user accounts, handling market data, implementing AI models, and ensuring secure data access.
3. **Defined the system's scope**:
   - Based on the requirements, I defined the scope of the backend application to ensure it would fully support the functionalities required for quantitative trading.
4. **Outlined initial resources and timelines**:
   - I determined the resources needed, such as development tools and technologies, and created a preliminary timeline for the development process.
5. **Designed the system architecture**:
   - A crucial step was designing a comprehensive database schema to ensure efficient data management. The schema was built with a focus on scalability and optimization to handle various data such as user information, market data, AI models, and trading strategies.

---

## Design <a name="design"></a>

1. **Database Design**:
   - Using the requirements I gathered, I designed a relational database schema in PostgreSQL. The schema includes several tables to efficiently manage users, market data, AI models, and trading strategies. With careful planning and normalization, I ensured data integrity and efficient storage.
2. **API Design**:
   - Next, I designed RESTful APIs to expose the necessary endpoints for the frontend application to interact with the data. I followed best practices for API design, including the use of HTTP methods, status codes, and sensible endpoint paths.
3. **Application Architecture Design**:
   - I adopted the MVC (Model-View-Controller) architecture for the application. With Express.js as the
