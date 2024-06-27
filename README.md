# Lumen

## Description <a name="description"></a>

Lumen is an advanced platform designed for quantitative trading strategies, incorporating AI-driven price prediction models and comprehensive user management systems. Inspired by institutional-grade practices from entities like Citadel, Lumen supports the development, training, and deployment of machine learning models to achieve precise market forecasting.

## Table of Contents <a name="table-of-contents"></a>

1. [Description](#description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [File & Directory Structure](#file-directory-structure)
   - [ai/](ai)
   - [api/](#api)
   - [db/](#db)
5. [APIs & Libraries Used](#api)
6. [Testing](#testing)
7. [Credits](#credits)
8. [Contact Information](#contact-information)

---

## Installation <a name="installation"></a>

To install and set up the Lumen platform, follow these steps:

1. **Clone the repository**:

   ```sh
   git clone https://github.com/your-username/lumen.git
   cd lumen
   ```

2. **Install dependencies**:

   ```sh
   npm install
   ```

3. **Set up environment variables**:
   Create a .env file in the root directory and add your configuration settings as shown below:

```env
DATABASE_URL=your_database_url
JWT_SECRET=your_jwt_secret
FINNHUB_API_KEY=your_finnhub_api_key
```

4. **Start the development server**:

```sh
npm run dev
```
