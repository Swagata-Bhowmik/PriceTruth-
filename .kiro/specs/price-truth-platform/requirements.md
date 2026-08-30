# Requirements Document

## Introduction

Price Truth is an AI-powered web platform that brings pricing transparency to Indian e-commerce. It addresses two consumer problems: **inflated "before" prices** that make discounts appear larger than they are, and **shrinkflation**, where FMCG pack sizes shrink over time while the price stays the same. The platform lets a shopper look up a product and receive an explainable, data-grounded read on whether a deal is real, how a product's pack size has changed, which pack size offers the best true value, whether the category tends to get cheaper at a known time of year, and how the price compares across major Indian platforms.

This document specifies the requirements for the working product built on top of the existing Phase 0 scaffolding. The scope is fixed to five features (True Discount Checker, Shrinkflation Timeline, Unit Price Comparator, Buy Timing Signal, Cross-Platform Aggregator), plus a product search entry point, a cohesive dashboard, SHAP-based explainability, graceful handling of incomplete data, and honest in-product disclosure of limitations.

The platform is built under three fixed constraints that shape every requirement: **zero budget** (only free tools and free tiers), **no manual data collection** (all data is sourced automatically from Kaggle datasets and the Open Food Facts public API), and **snapshot data** (public datasets are point-in-time snapshots rather than daily per-product time series). Because of the snapshot constraint, buy-timing predictions are deliberately made at the **category level** and this limitation is disclosed openly rather than hidden.

Non-Functional Requirements are treated as first-class and graded. This document includes a dedicated, testable set of Non-Functional Requirements mapped to nine categories: Performance, Scalability, Portability, Compatibility, Reliability, Availability, Maintainability, Security, and Usability.

## Glossary

- **Price_Truth_Platform**: The complete web application, comprising the React frontend, the FastAPI backend, the machine learning layer, and the data layer.
- **Product_Search**: The entry-point component that accepts a product query and routes the resolved product into the five feature modules.
- **Discount_Checker**: The component that classifies a displayed discount as genuine or inflated and returns a genuineness score.
- **SHAP_Explainer**: The component that produces a feature-by-feature contribution breakdown explaining a Discount_Checker result.
- **Shrinkflation_Timeline**: The component that presents documented pack-size changes for an FMCG product over time.
- **Unit_Price_Comparator**: The component that computes and compares price per standard unit (per gram or per milliliter) across pack sizes and brands.
- **Buy_Timing_Analyzer**: The component that produces a category-level buy-now-or-wait recommendation from seasonal price patterns.
- **Cross_Platform_Aggregator**: The component that compares a product's price across supported e-commerce platforms and identifies the lowest-price option.
- **Dashboard**: The single cohesive interface that presents all five feature modules to the user.
- **Data_Service**: The backend layer responsible for ingesting Kaggle datasets, calling the Open Food Facts API, caching results, and serving cleaned data to feature modules.
- **Discount Authenticity**: The degree to which a displayed discount reflects a product's real recent price history rather than an inflated reference price.
- **Genuineness Score**: A value from 0 to 100 percent expressing the modeled likelihood that a displayed discount is genuine.
- **Inflated Discount**: A discount presented against a reference ("original") price that is higher than the product's real recent selling price.
- **Shrinkflation**: A reduction in a product's pack quantity while the selling price is held constant, increasing the effective unit price.
- **Unit Price**: The price of a product expressed per standard unit of quantity, specifically per gram for solids and per milliliter for liquids.
- **SHAP**: SHapley Additive exPlanations, a method that attributes a model prediction to the contribution of each input feature relative to a baseline value.
- **Category-Level Prediction**: A prediction expressed for a product category rather than for a single product on a single future date.
- **Snapshot Data**: Point-in-time price records that do not form a continuous daily time series for a single product.
- **Open Food Facts**: A free, public, crowd-sourced product database accessed through a public API, providing product name, brand, quantity, and category.
- **FMCG**: Fast-Moving Consumer Goods, such as packaged food and household products.
- **Supported Platform**: One of the e-commerce platforms compared by the Cross_Platform_Aggregator: Amazon, Flipkart, Croma, Tata CLiQ, and Reliance Digital.
- **Cache Hit**: A request served from the Redis cache without recomputation or an external API call.
- **EARS**: Easy Approach to Requirements Syntax, the pattern set used for the acceptance criteria in this document.

## Requirements

### Requirement 1: Product Search and Lookup

**User Story:** As Priya, a smart online shopper, I want to look up a product by name or link, so that I can feed it into the platform's analysis features without manual data entry.

#### Acceptance Criteria

1. WHEN a user submits a non-empty product query, THE Product_Search SHALL return a list of matching products within 3 seconds.
2. WHEN the Product_Search returns matching products, THE Product_Search SHALL display for each match the product name, brand, and category.
3. WHEN a user selects a matched product, THE Price_Truth_Platform SHALL make the selected product available to the Discount_Checker, Shrinkflation_Timeline, Unit_Price_Comparator, Buy_Timing_Analyzer, and Cross_Platform_Aggregator.
4. IF a submitted query is empty, THEN THE Product_Search SHALL display a message prompting the user to enter a product name.
5. IF a submitted query returns no matches, THEN THE Product_Search SHALL display a no-results message and offer a manual-entry option for price and pack details.
6. WHERE a user enters a product manually, THE Product_Search SHALL accept a product name, a displayed price, a reference price, and a pack quantity as inputs.

### Requirement 2: True Discount Checker

**User Story:** As Aarav, a price-sensitive student, I want to know whether a displayed discount is genuine, so that I can avoid being misled by inflated "original" prices.

#### Acceptance Criteria

1. WHEN a user requests a discount check for a product with a displayed price and a reference price, THE Discount_Checker SHALL return a Genuineness Score between 0 and 100 percent.
2. WHEN the Discount_Checker returns a Genuineness Score, THE Discount_Checker SHALL classify the discount as genuine WHERE the score is at least 90 percent, as moderate WHERE the score is at least 60 percent and below 90 percent, and as likely inflated WHERE the score is below 60 percent.
3. WHEN the Discount_Checker computes a Genuineness Score, THE Discount_Checker SHALL derive the score from the trained XGBoost model using the product's category price statistics as input features.
4. WHEN the Discount_Checker returns a result, THE Discount_Checker SHALL display the displayed price, the reference price, and the effective discount percentage alongside the classification.
5. IF the reference price is missing or is less than or equal to the displayed price, THEN THE Discount_Checker SHALL report that a discount cannot be evaluated and SHALL state the reason.
6. IF category price statistics are unavailable for the selected product, THEN THE Discount_Checker SHALL report that verification is limited and SHALL present available price context without a Genuineness Score.

### Requirement 3: SHAP Explainability

**User Story:** As Priya, I want to see why a discount was judged genuine or inflated, so that I can trust the verdict instead of accepting a black-box score.

#### Acceptance Criteria

1. WHEN the Discount_Checker returns a Genuineness Score, THE SHAP_Explainer SHALL produce a feature-by-feature contribution breakdown for that result.
2. WHEN the SHAP_Explainer produces a contribution breakdown, THE SHAP_Explainer SHALL display each contributing feature, the magnitude of its contribution, and whether the contribution moves the result toward genuine or toward inflated.
3. WHEN the SHAP_Explainer displays a contribution breakdown, THE SHAP_Explainer SHALL present the baseline value and the final score so that the contributions reconcile to the result.
4. WHEN the SHAP_Explainer computes contributions, THE SHAP_Explainer SHALL derive contributions from the same trained model instance that produced the Genuineness Score.
5. THE SHAP_Explainer SHALL label each displayed feature with a plain-language name rather than a raw model feature identifier.

### Requirement 4: Shrinkflation Timeline

**User Story:** As Rajesh, a budget-conscious grocery buyer, I want to see how a product's pack size has changed over time at the same price, so that I can understand hidden unit-price increases.

#### Acceptance Criteria

1. WHEN a user selects an FMCG product that has recorded pack-size history, THE Shrinkflation_Timeline SHALL display the pack quantity and selling price for each recorded time point in chronological order.
2. WHEN the Shrinkflation_Timeline displays pack-size history, THE Shrinkflation_Timeline SHALL compute and display the Unit Price at each recorded time point.
3. WHEN pack-size history spans two or more time points, THE Shrinkflation_Timeline SHALL display the total percentage change in pack quantity and the total percentage change in Unit Price across the full recorded period.
4. WHEN the Shrinkflation_Timeline displays a data point sourced from Open Food Facts or a cited public record, THE Shrinkflation_Timeline SHALL display the source attribution for that data point.
5. IF a selected product has no recorded pack-size history, THEN THE Shrinkflation_Timeline SHALL display a message stating that pack-size history is unavailable for the product.

### Requirement 5: Unit Price Comparator

**User Story:** As Rajesh, I want to compare the true per-unit cost of different pack sizes and brands, so that I can identify the best real value instead of assuming the largest pack is cheapest.

#### Acceptance Criteria

1. WHEN a user provides two or more product variants with a price and a pack quantity, THE Unit_Price_Comparator SHALL compute the Unit Price for each variant.
2. WHEN the Unit_Price_Comparator computes Unit Prices for a set of variants, THE Unit_Price_Comparator SHALL identify the variant with the lowest Unit Price as the best value.
3. WHEN the Unit_Price_Comparator displays a comparison, THE Unit_Price_Comparator SHALL display each variant's price, pack quantity, and computed Unit Price in a single comparison view.
4. WHERE variants are expressed in different units of the same measure, THE Unit_Price_Comparator SHALL convert each pack quantity to a common standard unit before computing Unit Prices.
5. IF a variant has a missing or non-positive pack quantity, THEN THE Unit_Price_Comparator SHALL exclude that variant from the comparison and SHALL indicate the exclusion.

### Requirement 6: Buy Timing Signal

**User Story:** As Aarav, I want guidance on whether to buy now or wait, so that I can time purchases to seasonal price drops within my limited budget.

#### Acceptance Criteria

1. WHEN a user requests buy-timing guidance for a product category, THE Buy_Timing_Analyzer SHALL return a recommendation of buy now or wait for that category.
2. WHEN the Buy_Timing_Analyzer returns a recommendation to wait, THE Buy_Timing_Analyzer SHALL display the seasonal window during which the category has historically shown its largest price reductions.
3. THE Buy_Timing_Analyzer SHALL express every recommendation at the category level rather than for a single product on a single future date.
4. WHEN the Buy_Timing_Analyzer displays a recommendation, THE Buy_Timing_Analyzer SHALL display a statement that the recommendation is category-level and is derived from snapshot data.
5. WHEN the Buy_Timing_Analyzer evaluates seasonality, THE Buy_Timing_Analyzer SHALL reference the Indian sale calendar dates for Big Billion Days, Republic Day Sale, Diwali, and Prime Day.
6. IF no seasonal pattern is available for a requested category, THEN THE Buy_Timing_Analyzer SHALL state that a timing recommendation is unavailable for the category.

### Requirement 7: Cross-Platform Aggregator

**User Story:** As Priya, I want to compare a product's price across major Indian platforms, so that I can choose the best overall deal in one place.

#### Acceptance Criteria

1. WHEN a user requests a cross-platform comparison for a product, THE Cross_Platform_Aggregator SHALL display the available price for the product on each Supported Platform for which data exists.
2. WHEN the Cross_Platform_Aggregator displays two or more platform prices, THE Cross_Platform_Aggregator SHALL identify the platform with the lowest price as the best deal.
3. WHEN the Cross_Platform_Aggregator displays a platform entry, THE Cross_Platform_Aggregator SHALL provide a link to the product page on that Supported Platform.
4. WHERE a Genuineness Score exists for a platform's listing, THE Cross_Platform_Aggregator SHALL display the Genuineness Score next to that platform entry.
5. IF price data exists for only one Supported Platform, THEN THE Cross_Platform_Aggregator SHALL display the single available price and SHALL state that no comparison is available.
6. IF price data exists for no Supported Platform, THEN THE Cross_Platform_Aggregator SHALL display a message stating that cross-platform data is unavailable for the product.

### Requirement 8: Cohesive Dashboard

**User Story:** As Priya, I want all five features presented together in one clean, compact view, so that I can understand a product's pricing truth at a glance.

#### Acceptance Criteria

1. WHEN a user opens the analysis for a selected product, THE Dashboard SHALL present the Discount_Checker, Shrinkflation_Timeline, Unit_Price_Comparator, Buy_Timing_Analyzer, and Cross_Platform_Aggregator within a single view.
2. WHEN the Dashboard renders on a viewport at least 1024 pixels wide, THE Dashboard SHALL present the primary result of each feature module without requiring the user to scroll past one screen height.
3. WHEN a user hovers over or activates a feature summary, THE Dashboard SHALL expand that feature to reveal its detailed view.
4. WHILE a feature module is computing a result, THE Dashboard SHALL display a loading indicator for that module.
5. IF a feature module returns no result for the selected product, THEN THE Dashboard SHALL display the module's unavailable-data message in place of the module result.

### Requirement 9: Graceful Handling of Incomplete Data

**User Story:** As any user, I want the platform to behave predictably when product data is missing, so that I still get useful partial information instead of an error.

#### Acceptance Criteria

1. WHEN the Data_Service receives a response from Open Food Facts with one or more missing fields, THE Data_Service SHALL return the available fields and SHALL mark each missing field as unavailable.
2. IF a call to the Open Food Facts API fails or exceeds a 5-second timeout, THEN THE Data_Service SHALL return a cached result WHERE a Cache Hit exists, and SHALL otherwise return a data-unavailable status.
3. WHEN a feature module receives a data-unavailable status for a required input, THE feature module SHALL display a message identifying the missing data.
4. WHEN the Data_Service serves a product previously retrieved within the cache validity period, THE Data_Service SHALL serve the product from the Redis cache.
5. THE Data_Service SHALL validate every external data value against its expected type and range before passing the value to a feature module.

### Requirement 10: Honest Disclosure of Limitations

**User Story:** As a professor or recruiter evaluating the platform, I want the product to state its data and prediction limitations openly, so that I can trust its integrity.

#### Acceptance Criteria

1. WHERE the Buy_Timing_Analyzer displays a recommendation, THE Price_Truth_Platform SHALL disclose that predictions are category-level and are based on snapshot data.
2. THE Price_Truth_Platform SHALL provide an accessible section that describes the platform's data sources and their known limitations.
3. WHEN the Price_Truth_Platform displays a result derived from crowd-sourced Open Food Facts data, THE Price_Truth_Platform SHALL indicate that the data is crowd-sourced and may be incomplete.
4. THE Price_Truth_Platform SHALL state that live scraping of Amazon and Flipkart is not used as a core data source.

## Non-Functional Requirements

### Requirement 11: Performance

**User Story:** As any user, I want the platform to respond quickly, so that I can check pricing truth without waiting.

#### Acceptance Criteria

1. WHEN a request results in a Cache Hit, THE Price_Truth_Platform SHALL return the response with a 95th-percentile latency of 500 milliseconds or less under a load of 50 concurrent users.
2. WHEN a user requests a discount check that requires model inference, THE Discount_Checker SHALL return the Genuineness Score within 3 seconds under warm-service conditions.
3. WHEN the Discount_Checker returns a Genuineness Score, THE SHAP_Explainer SHALL return the contribution breakdown within an additional 2 seconds.
4. WHEN the Dashboard loads for a selected product on a 10 Mbps connection, THE Dashboard SHALL reach first contentful paint within 3 seconds.

### Requirement 12: Scalability

**User Story:** As the platform owner, I want the system to handle growing usage on free-tier infrastructure, so that traffic increases do not cause failures.

#### Acceptance Criteria

1. WHILE serving up to 50 concurrent users, THE Price_Truth_Platform SHALL maintain an error rate at or below 1 percent of requests.
2. THE Price_Truth_Platform SHALL expose the backend as stateless request handlers so that request handling does not depend on in-process session state.
3. WHEN repeated requests are made for the same product within the cache validity period, THE Data_Service SHALL serve the requests from the Redis cache rather than recomputing results or repeating external API calls.
4. WHERE model inference is required, THE Discount_Checker SHALL load the trained model once per process and SHALL reuse the loaded model across requests.

### Requirement 13: Portability

**User Story:** As a developer, I want the platform to run consistently across environments, so that it can be deployed on free tiers and on a local machine without code changes.

#### Acceptance Criteria

1. THE Price_Truth_Platform SHALL read all environment-specific configuration, including database connection details and API endpoints, from environment variables.
2. THE Price_Truth_Platform SHALL provide container definitions that allow the backend and its dependencies to run through a single orchestration command.
3. THE Price_Truth_Platform SHALL deploy the frontend to the Vercel free tier and the backend and database to the Railway free tier without paid dependencies.
4. THE Price_Truth_Platform SHALL exclude credentials and secrets from the source repository.

### Requirement 14: Compatibility

**User Story:** As any user, I want the platform to work on my browser and device, so that I can use it regardless of screen size.

#### Acceptance Criteria

1. WHEN the Dashboard is opened in the current or immediately previous major version of Chrome, Firefox, Safari, or Edge, THE Dashboard SHALL render all five feature modules without layout overflow.
2. WHEN the Dashboard renders on a viewport width of 480 pixels or less, THE Dashboard SHALL present its content in a single-column layout.
3. WHEN the Dashboard renders on a viewport width between 481 and 1023 pixels, THE Dashboard SHALL adapt its layout for tablet display without horizontal scrolling.
4. THE Price_Truth_Platform SHALL expose backend responses in JSON so that any standards-compliant HTTP client can consume the responses.

### Requirement 15: Reliability

**User Story:** As any user, I want the platform to fail safely, so that one broken input or dependency does not break the whole experience.

#### Acceptance Criteria

1. IF a feature module raises an unhandled error, THEN THE Price_Truth_Platform SHALL contain the error within that module and SHALL continue to serve the remaining feature modules.
2. IF an external API call fails, THEN THE Data_Service SHALL retry the call at most 2 times before returning a data-unavailable status.
3. WHEN THE Price_Truth_Platform returns an error response, THE Price_Truth_Platform SHALL return a structured error payload containing a human-readable message and a status code.
4. THE Data_Service SHALL reject any external data value that fails type or range validation and SHALL record the rejection in the application log.

### Requirement 16: Availability

**User Story:** As a user or evaluator, I want the platform to be reachable when I use it, so that I can rely on it during the demo and evaluation period.

#### Acceptance Criteria

1. THE Price_Truth_Platform SHALL expose a health-check endpoint that returns a success status when the backend and database connections are operational.
2. THE Price_Truth_Platform SHALL target a monthly availability of 99 percent during evaluation periods, excluding free-tier cold-start delays.
3. WHEN the backend resumes from a free-tier cold start, THE Price_Truth_Platform SHALL become ready to serve requests within 30 seconds.
4. WHILE the database is unreachable, THE Price_Truth_Platform SHALL return a service-unavailable status with a retry message rather than an unhandled failure.

### Requirement 17: Maintainability

**User Story:** As a developer continuing this project, I want a well-tested and documented codebase, so that I can extend it confidently and demonstrate code quality.

#### Acceptance Criteria

1. THE Price_Truth_Platform SHALL include automated tests covering the discount classification logic, the Unit Price computation, and the data-validation logic.
2. THE Price_Truth_Platform SHALL achieve automated test coverage of at least 70 percent for backend business-logic modules.
3. THE Price_Truth_Platform SHALL pass a configured linter for Python code and a configured linter for JavaScript code without reported errors.
4. THE Price_Truth_Platform SHALL publish interactive API documentation generated from the backend service definitions.
5. THE Price_Truth_Platform SHALL organize backend code into separate modules for API routing, data services, and machine learning so that each concern is modifiable in isolation.

### Requirement 18: Security

**User Story:** As any user, I want my inputs handled safely, so that using the platform does not expose me or the system to attacks.

#### Acceptance Criteria

1. WHEN THE Price_Truth_Platform receives a request, THE Price_Truth_Platform SHALL validate every input parameter against its expected type, length, and format before processing.
2. WHEN THE Data_Service accesses the database, THE Data_Service SHALL use parameterized queries through the object-relational mapper so that user input is not concatenated into query strings.
3. THE Price_Truth_Platform SHALL restrict cross-origin requests to the configured frontend origin.
4. IF a single client exceeds 60 requests per minute, THEN THE Price_Truth_Platform SHALL reject additional requests from that client with a rate-limit status until the next minute begins.
5. THE Price_Truth_Platform SHALL serve all client traffic over HTTPS.
6. THE Price_Truth_Platform SHALL load all credentials from environment variables and SHALL exclude credentials from source control and from client responses.

### Requirement 19: Usability

**User Story:** As Priya, Rajesh, or Aarav, I want a clear, readable, and accessible interface, so that I can understand pricing truth without confusion.

#### Acceptance Criteria

1. THE Dashboard SHALL present text and interactive controls with a color contrast ratio of at least 4.5 to 1 for normal text and at least 3 to 1 for large text.
2. WHEN the Discount_Checker displays a classification, THE Dashboard SHALL convey the classification through a text label in addition to color.
3. WHEN a user activates any interactive control using a keyboard, THE Dashboard SHALL provide a visible focus indicator for that control.
4. WHEN a feature module displays a result, THE Dashboard SHALL present the module's primary conclusion before its supporting detail.
5. THE Dashboard SHALL provide descriptive text alternatives for every chart and non-text visual element.
