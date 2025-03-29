import os
import json
import pandas as pd
from dotenv import load_dotenv
import pyodbc
import sqlalchemy as db
from sqlalchemy import text, inspect
from groq import Groq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

GROQ_API_KEY = "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"

DB_CONNECTION_STRING = "mssql+pyodbc://@localhost/AdventureWorksLT2022?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"

console = Console()

class SQLDatabaseRAG:
    def __init__(self, connection_string=DB_CONNECTION_STRING, groq_api_key=GROQ_API_KEY):
        """Initialize the SQL Database RAG system."""
        self.connection_string = connection_string
        self.groq_client = Groq(api_key=groq_api_key)
        self.engine = None
        self.conversation_history = []
        self.schema_info = None
        
        # Try to connect to the database
        try:
            self.engine = db.create_engine(connection_string)
            self.connection = self.engine.connect()
            print("Successfully connected to the database.")
            # Extract schema directly from the database
            self.schema_info = self._extract_schema_from_db()
        except Exception as e:
            print(f"Error connecting to database: {e}")
            print("Will load static schema information instead.")
            self.schema_info = self._get_static_schema_description()
    
    def _extract_schema_from_db(self):
        """
        Extract schema information directly from the connected database.
        This provides complete and accurate schema information.
        """
        if not self.engine:
            return self._get_static_schema_description()
        
        try:
            inspector = inspect(self.engine)
            schema_info = "AdventureWorks Database Schema:\n\n"
            
            # Get all schemas
            query = text("SELECT DISTINCT TABLE_SCHEMA FROM INFORMATION_SCHEMA.TABLES ORDER BY TABLE_SCHEMA")
            schemas = [row[0] for row in self.connection.execute(query)]
            
            for schema in schemas:
                # Get all tables for this schema
                tables = inspector.get_table_names(schema=schema)
                
                for table_name in tables:
                    full_table_name = f"{schema}.{table_name}"
                    schema_info += f"Table: {full_table_name}\n"
                    
                    # Get table description if available (from sys.extended_properties)
                    desc_query = text(f"""
                    SELECT ep.value 
                    FROM sys.extended_properties ep
                    JOIN sys.tables t ON ep.major_id = t.object_id
                    JOIN sys.schemas s ON t.schema_id = s.schema_id
                    WHERE ep.name = 'MS_Description'
                    AND ep.minor_id = 0
                    AND s.name = '{schema}'
                    AND t.name = '{table_name}'
                    """)
                    
                    description_result = self.connection.execute(desc_query).fetchone()
                    table_description = description_result[0] if description_result else "No description available"
                    schema_info += f"Description: {table_description}\n"
                    
                    # Get columns and their info
                    schema_info += "Columns:\n"
                    columns = inspector.get_columns(table_name, schema=schema)
                    
                    for column in columns:
                        col_name = column['name']
                        col_type = str(column['type'])
                        
                        # Get column description if available
                        col_desc_query = text(f"""
                        SELECT ep.value 
                        FROM sys.extended_properties ep
                        JOIN sys.tables t ON ep.major_id = t.object_id
                        JOIN sys.columns c ON ep.major_id = c.object_id AND ep.minor_id = c.column_id
                        JOIN sys.schemas s ON t.schema_id = s.schema_id
                        WHERE ep.name = 'MS_Description'
                        AND s.name = '{schema}'
                        AND t.name = '{table_name}'
                        AND c.name = '{col_name}'
                        """)
                        
                        col_desc_result = self.connection.execute(col_desc_query).fetchone()
                        col_description = col_desc_result[0] if col_desc_result else "No description available"
                        
                        schema_info += f"  - {col_name} ({col_type}): {col_description}\n"
                    
                    # Get primary key information
                    pk_constraint = inspector.get_pk_constraint(table_name, schema=schema)
                    if pk_constraint and 'constrained_columns' in pk_constraint:
                        pk_cols = ', '.join(pk_constraint['constrained_columns'])
                        schema_info += f"Primary Key: {pk_cols}\n"
                    
                    # Get foreign key information
                    fks = inspector.get_foreign_keys(table_name, schema=schema)
                    if fks:
                        schema_info += "Foreign Keys:\n"
                        for fk in fks:
                            referred_schema = fk.get('referred_schema')
                            referred_table = fk.get('referred_table')
                            referred_columns = fk.get('referred_columns')
                            constrained_columns = fk.get('constrained_columns')
                            
                            fk_info = f"  - {', '.join(constrained_columns)} -> "
                            if referred_schema:
                                fk_info += f"{referred_schema}."
                            fk_info += f"{referred_table}({', '.join(referred_columns)})"
                            schema_info += fk_info + "\n"
                    
                    schema_info += "\n"
            
            # Limit the schema information size if it's too large
            if len(schema_info) > 50000:  # Reasonable size to include in prompts
                console.print("[yellow]Schema is very large, including only the most important tables[/yellow]")
                # Include only the main sales-related tables if schema is too large
                important_schemas = ["Sales", "Production", "Person", "HumanResources"]
                important_tables = ["SalesOrderHeader", "SalesOrderDetail", "Product", "Customer", 
                                  "Person", "ProductCategory", "SalesTerritory"]
                
                filtered_schema = "AdventureWorks Database Schema (Important Tables Only):\n\n"
                
                for line in schema_info.split("\n"):
                    if line.startswith("Table:"):
                        table_name = line.replace("Table:", "").strip()
                        schema_name = table_name.split('.')[0] if '.' in table_name else ""
                        base_table_name = table_name.split('.')[-1] if '.' in table_name else table_name
                        
                        if (schema_name in important_schemas or 
                            any(imp_table in base_table_name for imp_table in important_tables)):
                            include_table = True
                        else:
                            include_table = False
                            
                    if include_table:
                        filtered_schema += line + "\n"
                        
                return filtered_schema
                        
            return schema_info
        except Exception as e:
            console.print(f"[red]Error extracting schema from database: {e}[/red]")
            return self._get_static_schema_description()
    
    def _get_static_schema_description(self):
        """
        Provide a static description of the AdventureWorks schema.
        This is a fallback if database connection is unavailable.
        """
        schema = {
            # Sales schema tables
            "sales_order_header": {
                "table_name": "Sales.SalesOrderHeader",
                "description": "Contains sales order header data including customer ID, order date, and total amount due.",
                "columns": [
                    {"name": "SalesOrderID", "type": "int", "description": "Primary key for SalesOrderHeader records."},
                    {"name": "RevisionNumber", "type": "tinyint", "description": "Incremental number to track changes to the sales order."},
                    {"name": "OrderDate", "type": "datetime", "description": "Date the order was placed."},
                    {"name": "DueDate", "type": "datetime", "description": "Date the order is due to the customer."},
                    {"name": "ShipDate", "type": "datetime", "description": "Date the order was shipped to the customer."},
                    {"name": "Status", "type": "tinyint", "description": "Order current status: 1 = In process, 2 = Approved, 3 = Backordered, 4 = Rejected, 5 = Shipped, 6 = Cancelled"},
                    {"name": "OnlineOrderFlag", "type": "bit", "description": "0 = Order placed by sales person, 1 = Order placed online."},
                    {"name": "SalesOrderNumber", "type": "nvarchar", "description": "Unique sales order identification number."},
                    {"name": "PurchaseOrderNumber", "type": "nvarchar", "description": "Customer purchase order number reference."},
                    {"name": "CustomerID", "type": "int", "description": "Foreign key to Customer.CustomerID."},
                    {"name": "TerritoryID", "type": "int", "description": "Territory in which the sale was made."},
                    {"name": "SubTotal", "type": "money", "description": "Sales subtotal. Computed as SUM(SalesOrderDetail.LineTotal)."},
                    {"name": "TaxAmt", "type": "money", "description": "Tax amount."},
                    {"name": "Freight", "type": "money", "description": "Shipping cost."},
                    {"name": "TotalDue", "type": "money", "description": "Total due from customer. Computed as Subtotal + TaxAmt + Freight."}
                ]
            },
            "sales_order_detail": {
                "table_name": "Sales.SalesOrderDetail",
                "description": "Contains individual line items associated with sales orders.",
                "columns": [
                    {"name": "SalesOrderID", "type": "int", "description": "Primary key. Foreign key to SalesOrderHeader.SalesOrderID."},
                    {"name": "SalesOrderDetailID", "type": "int", "description": "Primary key. One increment per product sold."},
                    {"name": "CarrierTrackingNumber", "type": "nvarchar", "description": "Shipment tracking number supplied by the shipper."},
                    {"name": "OrderQty", "type": "smallint", "description": "Quantity ordered per product."},
                    {"name": "ProductID", "type": "int", "description": "Foreign key to Product.ProductID."},
                    {"name": "SpecialOfferID", "type": "int", "description": "Foreign key to SpecialOffer.SpecialOfferID."},
                    {"name": "UnitPrice", "type": "money", "description": "Selling price of a single product."},
                    {"name": "UnitPriceDiscount", "type": "money", "description": "Discount amount."},
                    {"name": "LineTotal", "type": "numeric", "description": "Per product subtotal. Computed as UnitPrice * (1 - UnitPriceDiscount) * OrderQty."}
                ]
            },
            "customer": {
                "table_name": "Sales.Customer",
                "description": "Current customer information.",
                "columns": [
                    {"name": "CustomerID", "type": "int", "description": "Primary key for Customer records."},
                    {"name": "PersonID", "type": "int", "description": "Foreign key to Person.Person (demographic data)."},
                    {"name": "StoreID", "type": "int", "description": "Foreign key to Store.Store (resellers)."},
                    {"name": "TerritoryID", "type": "int", "description": "Foreign key to SalesTerritory.SalesTerritoryID."},
                    {"name": "AccountNumber", "type": "varchar", "description": "Unique account identification number."}
                ]
            },
            "territory": {
                "table_name": "Sales.SalesTerritory",
                "description": "Sales territory lookup table.",
                "columns": [
                    {"name": "TerritoryID", "type": "int", "description": "Primary key for SalesTerritory records."},
                    {"name": "Name", "type": "nvarchar", "description": "Sales territory name."},
                    {"name": "CountryRegionCode", "type": "nvarchar", "description": "ISO standard country or region code."},
                    {"name": "Group", "type": "nvarchar", "description": "Geographic area to which the sales territory belongs."}
                ]
            },
            "special_offer": {
                "table_name": "Sales.SpecialOffer",
                "description": "Special offer information such as discounts and promotions.",
                "columns": [
                    {"name": "SpecialOfferID", "type": "int", "description": "Primary key for SpecialOffer records."},
                    {"name": "Description", "type": "nvarchar", "description": "Discount description."},
                    {"name": "DiscountPct", "type": "money", "description": "Discount percentage."},
                    {"name": "Type", "type": "nvarchar", "description": "Discount type such as Promotion or Seasonal Discount."},
                    {"name": "StartDate", "type": "datetime", "description": "Discount start date."},
                    {"name": "EndDate", "type": "datetime", "description": "Discount end date."}
                ]
            },
            "sales_person": {
                "table_name": "Sales.SalesPerson",
                "description": "Sales representative information.",
                "columns": [
                    {"name": "BusinessEntityID", "type": "int", "description": "Primary key. Foreign key to Employee.BusinessEntityID."},
                    {"name": "TerritoryID", "type": "int", "description": "Foreign key to SalesTerritory.SalesTerritoryID."},
                    {"name": "SalesQuota", "type": "money", "description": "Sales quota for the sales person."},
                    {"name": "Bonus", "type": "money", "description": "Current bonus."},
                    {"name": "CommissionPct", "type": "smallmoney", "description": "Commission percent."},
                    {"name": "SalesYTD", "type": "money", "description": "Sales year to date."},
                    {"name": "SalesLastYear", "type": "money", "description": "Sales last year."}
                ]
            },
            "store": {
                "table_name": "Sales.Store",
                "description": "Customers (resellers) of Adventure Works products.",
                "columns": [
                    {"name": "BusinessEntityID", "type": "int", "description": "Primary key. Foreign key to Customer.CustomerID."},
                    {"name": "Name", "type": "nvarchar", "description": "Store name."},
                    {"name": "SalesPersonID", "type": "int", "description": "Foreign key to SalesPerson.SalesPersonID."}
                ]
            },
            
            # Production schema tables
            "product": {
                "table_name": "Production.Product",
                "description": "Products sold or used in the manufacturing of sold products.",
                "columns": [
                    {"name": "ProductID", "type": "int", "description": "Primary key for Product records."},
                    {"name": "Name", "type": "nvarchar", "description": "Name of the product."},
                    {"name": "ProductNumber", "type": "nvarchar", "description": "Unique product identification number."},
                    {"name": "MakeFlag", "type": "bit", "description": "0 = Product is purchased, 1 = Product is manufactured in-house."},
                    {"name": "FinishedGoodsFlag", "type": "bit", "description": "0 = Product is not a salable item. 1 = Product is salable."},
                    {"name": "Color", "type": "nvarchar", "description": "Product color."},
                    {"name": "SafetyStockLevel", "type": "smallint", "description": "Minimum inventory quantity."},
                    {"name": "ReorderPoint", "type": "smallint", "description": "Inventory level that triggers a purchase order or work order."},
                    {"name": "StandardCost", "type": "money", "description": "Standard cost of the product."},
                    {"name": "ListPrice", "type": "money", "description": "Selling price."},
                    {"name": "Size", "type": "nvarchar", "description": "Product size."},
                    {"name": "ProductSubcategoryID", "type": "int", "description": "Foreign key to ProductSubcategory.ProductSubcategoryID."},
                    {"name": "ProductModelID", "type": "int", "description": "Foreign key to ProductModel.ProductModelID."}
                ]
            },
            "product_category": {
                "table_name": "Production.ProductCategory",
                "description": "Categories for products.",
                "columns": [
                    {"name": "ProductCategoryID", "type": "int", "description": "Primary key for ProductCategory records."},
                    {"name": "Name", "type": "nvarchar", "description": "Category name."},
                    {"name": "ParentProductCategoryID", "type": "int", "description": "Foreign key to ProductCategory.ProductCategoryID."}
                ]
            },
            "product_subcategory": {
                "table_name": "Production.ProductSubcategory",
                "description": "Product subcategories, which are children of the ProductCategory table.",
                "columns": [
                    {"name": "ProductSubcategoryID", "type": "int", "description": "Primary key for ProductSubcategory records."},
                    {"name": "ProductCategoryID", "type": "int", "description": "Foreign key to ProductCategory.ProductCategoryID."},
                    {"name": "Name", "type": "nvarchar", "description": "Subcategory name."}
                ]
            },
            "product_inventory": {
                "table_name": "Production.ProductInventory",
                "description": "Product inventory information.",
                "columns": [
                    {"name": "ProductID", "type": "int", "description": "Primary key. Foreign key to Product.ProductID."},
                    {"name": "LocationID", "type": "int", "description": "Primary key. Foreign key to Location.LocationID."},
                    {"name": "Shelf", "type": "nvarchar", "description": "Storage compartment within a location."},
                    {"name": "Bin", "type": "tinyint", "description": "Storage container on a shelf in a location."},
                    {"name": "Quantity", "type": "smallint", "description": "Quantity of products in a specific location."}
                ]
            },
            
            # Person schema tables
            "person": {
                "table_name": "Person.Person",
                "description": "Human beings associated with AdventureWorks as contacts, employees, or customers.",
                "columns": [
                    {"name": "BusinessEntityID", "type": "int", "description": "Primary key for Person records."},
                    {"name": "PersonType", "type": "nchar", "description": "Primary type of person: SC = Store Contact, IN = Individual customer, SP = Sales person, etc."},
                    {"name": "FirstName", "type": "nvarchar", "description": "First name of the person."},
                    {"name": "LastName", "type": "nvarchar", "description": "Last name of the person."},
                    {"name": "EmailPromotion", "type": "int", "description": "0 = Contact does not wish to receive e-mail promotions, 1 = Contact does wish to receive e-mail promotions from AdventureWorks, 2 = Contact does wish to receive e-mail promotions from AdventureWorks and selected partners."}
                ]
            },
            "address": {
                "table_name": "Person.Address",
                "description": "Street address information.",
                "columns": [
                    {"name": "AddressID", "type": "int", "description": "Primary key for Address records."},
                    {"name": "AddressLine1", "type": "nvarchar", "description": "First street address line."},
                    {"name": "AddressLine2", "type": "nvarchar", "description": "Second street address line."},
                    {"name": "City", "type": "nvarchar", "description": "City name."},
                    {"name": "StateProvinceID", "type": "int", "description": "Foreign key to StateProvince table."},
                    {"name": "PostalCode", "type": "nvarchar", "description": "Postal code."}
                ]
            },
            
            # Human Resources schema tables
            "employee": {
                "table_name": "HumanResources.Employee",
                "description": "Employee information such as job title, hire date, and work history.",
                "columns": [
                    {"name": "BusinessEntityID", "type": "int", "description": "Primary key. Foreign key to Person.BusinessEntityID."},
                    {"name": "NationalIDNumber", "type": "nvarchar", "description": "Unique national identification number."},
                    {"name": "LoginID", "type": "nvarchar", "description": "Network login ID."},
                    {"name": "JobTitle", "type": "nvarchar", "description": "Work title such as Buyer or Sales Representative."},
                    {"name": "BirthDate", "type": "date", "description": "Date of birth."},
                    {"name": "MaritalStatus", "type": "nchar", "description": "M = Married, S = Single"},
                    {"name": "Gender", "type": "nchar", "description": "M = Male, F = Female"},
                    {"name": "HireDate", "type": "date", "description": "Employee hired on this date."},
                    {"name": "SalariedFlag", "type": "bit", "description": "0 = Paid by hour, 1 = Paid an annual salary"},
                    {"name": "VacationHours", "type": "smallint", "description": "Number of available vacation hours."},
                    {"name": "SickLeaveHours", "type": "smallint", "description": "Number of available sick leave hours."}
                ]
            },
            "department": {
                "table_name": "HumanResources.Department",
                "description": "Lookup table of departments within the Adventure Works organization.",
                "columns": [
                    {"name": "DepartmentID", "type": "smallint", "description": "Primary key for Department records."},
                    {"name": "Name", "type": "nvarchar", "description": "Department name."},
                    {"name": "GroupName", "type": "nvarchar", "description": "Name of the group to which the department belongs."}
                ]
            }
        }
        
        # Format schema for LLM consumption
        formatted_schema = "AdventureWorks Database Schema:\n\n"
        for table_id, table_info in schema.items():
            formatted_schema += f"Table: {table_info['table_name']}\n"
            formatted_schema += f"Description: {table_info['description']}\n"
            formatted_schema += "Columns:\n"
            for column in table_info['columns']:
                formatted_schema += f"  - {column['name']} ({column['type']}): {column['description']}\n"
            formatted_schema += "\n"
            
        return formatted_schema
    
    def _generate_sql(self, user_query):
        """
        Generate SQL query from natural language using Groq.
        """
        # Construct prompt with schema information and user query
        prompt = f"""You are an AI assistant that converts natural language questions about the AdventureWorks database into SQL queries.
        
Here is the database schema information:
{self.schema_info}

Recent conversation history:
{self._format_conversation_history()}

Based on the schema above and the conversation history, write a SQL query to answer this question: "{user_query}"

Return ONLY the SQL query without any explanation. The query should be correct, executable SQL for SQL Server.
"""
        
        # Get SQL query from Groq
        response = self.groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        # Extract SQL query
        sql_query = response.choices[0].message.content.strip()
        
        # Remove any markdown formatting if present
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        return sql_query
    
    def _execute_sql(self, sql_query):
        """
        Execute the generated SQL query and return the results.
        """
        if not self.engine:
            return "Database connection not available. Using schema information only."
        
        try:
            # Execute the query and fetch results
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                df = pd.DataFrame(result.fetchall())
                if not df.empty:
                    df.columns = result.keys()
                return df
        except Exception as e:
            return f"Error executing SQL query: {str(e)}"
    
    def _format_conversation_history(self):
        """Format conversation history for context."""
        if not self.conversation_history:
            return "No previous conversation."
        
        formatted_history = ""
        for i, (role, message) in enumerate(self.conversation_history[-5:]):  # Last 5 messages
            formatted_history += f"{role}: {message}\n"
        
        return formatted_history
    
    def _generate_response(self, user_query, sql_query, sql_result):
        """
        Generate a natural language response based on the SQL results.
        """
        # Format SQL results for the LLM
        if isinstance(sql_result, pd.DataFrame):
            if sql_result.empty:
                result_text = "The query returned no results."
            else:
                # Convert to string with reasonable formatting
                result_text = f"The query returned {len(sql_result)} rows. Here is a sample:\n"
                result_text += sql_result.head(10).to_string()
        else:
            result_text = str(sql_result)
        
        prompt = f"""You are an AI assistant that helps users understand data from the AdventureWorks database.

User query: "{user_query}"

SQL query executed:
```sql
{sql_query}
```

Query results:
{result_text}

Based on the SQL query and its results, provide a clear and concise answer to the user's question. 
If the results contain meaningful insights, mention them. 
If the query returned an error or no results, suggest a better approach if possible.
"""
        
        response = self.groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def process_query(self, user_query):
        """
        Process a user query end-to-end.
        1. Generate SQL from natural language
        2. Execute the SQL query
        3. Generate response based on results
        4. Update conversation history
        """
        # Store user query in conversation history
        self.conversation_history.append(("User", user_query))
        
        # Step 1: Generate SQL
        console.print(Panel("🔍 Generating SQL query...", title="Processing"))
        sql_query = self._generate_sql(user_query)
        console.print(Panel(f"```sql\n{sql_query}\n```", title="Generated SQL Query"))
        
        # Step 2: Execute SQL
        console.print(Panel("⚙️ Executing query...", title="Processing"))
        result = self._execute_sql(sql_query)
        
        # Step 3: Generate response
        console.print(Panel("🤖 Generating response...", title="Processing"))
        response = self._generate_response(user_query, sql_query, result)
        
        # Step 4: Update conversation history
        self.conversation_history.append(("Assistant", response))
        
        return {
            "user_query": user_query,
            "sql_query": sql_query,
            "sql_result": result,
            "response": response
        }

    def predict_future_sales(self, product_category=None, prediction_months=3):
        """
        Make sales predictions based on historical data.
        """
        # SQL to get historical sales data by month and category
        if product_category:
            # For a specific product category
            sql_query = f"""
            SELECT 
                YEAR(soh.OrderDate) as Year,
                MONTH(soh.OrderDate) as Month,
                pc.Name as CategoryName,
                SUM(sod.OrderQty * sod.UnitPrice) as TotalSales
            FROM 
                Sales.SalesOrderHeader soh
                JOIN Sales.SalesOrderDetail sod ON soh.SalesOrderID = sod.SalesOrderID
                JOIN Production.Product p ON sod.ProductID = p.ProductID
                JOIN Production.ProductSubcategory psc ON p.ProductSubcategoryID = psc.ProductSubcategoryID
                JOIN Production.ProductCategory pc ON psc.ProductCategoryID = pc.ProductCategoryID
            WHERE 
                pc.Name = '{product_category}'
            GROUP BY 
                YEAR(soh.OrderDate),
                MONTH(soh.OrderDate),
                pc.Name
            ORDER BY 
                Year, Month
            """
        else:
            # For all product categories
            sql_query = """
            SELECT 
                YEAR(soh.OrderDate) as Year,
                MONTH(soh.OrderDate) as Month,
                SUM(sod.OrderQty * sod.UnitPrice) as TotalSales
            FROM 
                Sales.SalesOrderHeader soh
                JOIN Sales.SalesOrderDetail sod ON soh.SalesOrderID = sod.SalesOrderID
            GROUP BY 
                YEAR(soh.OrderDate),
                MONTH(soh.OrderDate)
            ORDER BY 
                Year, Month
            """
        
        try:
            # Execute the query
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                historical_sales = pd.DataFrame(result.fetchall())
                if not historical_sales.empty:
                    historical_sales.columns = result.keys()
                
                # Simple time series forecasting using historical patterns
                if not historical_sales.empty:
                    # Create time feature and sort by date
                    historical_sales['Date'] = pd.to_datetime(historical_sales[['Year', 'Month']].assign(DAY=1))
                    historical_sales = historical_sales.sort_values('Date')
                    
                    # Calculate average growth rate
                    historical_sales['Growth'] = historical_sales['TotalSales'].pct_change()
                    avg_growth = historical_sales['Growth'].mean()
                    
                    # Get last date and sales value
                    last_date = historical_sales['Date'].max()
                    last_sales = historical_sales.loc[historical_sales['Date'] == last_date, 'TotalSales'].values[0]
                    
                    # Generate predictions
                    predictions = []
                    current_sales = last_sales
                    
                    for i in range(1, prediction_months + 1):
                        next_date = last_date + pd.DateOffset(months=i)
                        next_sales = current_sales * (1 + avg_growth)
                        predictions.append({
                            'Year': next_date.year,
                            'Month': next_date.month,
                            'PredictedSales': next_sales
                        })
                        current_sales = next_sales
                    
                    predictions_df = pd.DataFrame(predictions)
                    return {"historical": historical_sales, "predictions": predictions_df}
                
                return "No historical sales data found for prediction."
        except Exception as e:
            return f"Error executing prediction query: {str(e)}"
    
    def run_interactive_session(self):
        """
        Run an interactive session in the console.
        """
        console.print(Panel(
            "[bold green]Comprehensive SQL RAG System for AdventureWorks[/bold green]\n\n"
            "Ask questions about sales, products, customers, and more.\n"
            "Type 'exit' to quit."
        ))
        
        while True:
            user_query = console.input("[bold blue]Ask a question:[/bold blue] ")
            
            if user_query.lower() in ('exit', 'quit'):
                break
                
            if "predict" in user_query.lower() or "forecast" in user_query.lower():
                # Handle prediction queries
                console.print(Panel("🔮 Generating sales prediction...", title="AI Prediction"))
                
                # Extract product category if mentioned
                product_category = None
                if "category" in user_query.lower():
                    # Try to extract category from query
                    # Simple extraction - in real system would use entity extraction
                    category_words = ["bikes", "components", "clothing", "accessories"]
                    for category in category_words:
                        if category in user_query.lower():
                            product_category = category.capitalize()
                            break
                
                prediction_result = self.predict_future_sales(product_category)
                
                if isinstance(prediction_result, dict):
                    # Format prediction result for display
                    prediction_text = "Historical sales data:\n"
                    prediction_text += prediction_result["historical"].tail(5).to_string() + "\n\n"
                    prediction_text += "Sales predictions for the next few months:\n"
                    prediction_text += prediction_result["predictions"].to_string()
                    
                    # Generate explanation with Groq
                    prompt = f"""You are an AI assistant that explains sales predictions.

Sales prediction data:
{prediction_text}

Product Category: {product_category if product_category else "All categories"}

Provide a brief explanation of these sales predictions. Label your response as an AI prediction.
"""
                    
                    response = self.groq_client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=1000
                    )
                    
                    explanation = response.choices[0].message.content
                    console.print(Panel(explanation, title="[bold purple]AI Predicted Response[/bold purple]"))
                else:
                    console.print(Panel(str(prediction_result), title="[bold red]Prediction Error[/bold red]"))
            else:
                # Handle regular queries
                result = self.process_query(user_query)
                console.print(Panel(result["response"], title="Answer"))

    def get_table_relationships(self):
        """
        Generate a visualization of table relationships for key tables.
        """
        if not self.engine:
            return "Database connection required to extract relationships."
            
        try:
            # Get foreign key relationships
            inspector = inspect(self.engine)
            relationships = []
            
            # Focus on important schemas
            important_schemas = ["Sales", "Production", "Person", "HumanResources"]
            
            for schema in important_schemas:
                tables = inspector.get_table_names(schema=schema)
                
                for table in tables:
                    fks = inspector.get_foreign_keys(table, schema=schema)
                    if fks:
                        for fk in fks:
                            ref_schema = fk.get('referred_schema', schema)
                            relationships.append({
                                'from_table': f"{schema}.{table}",
                                'to_table': f"{ref_schema}.{fk['referred_table']}",
                                'from_column': ', '.join(fk['constrained_columns']),
                                'to_column': ', '.join(fk['referred_columns'])
                            })
            
            # Create a table of relationships
            rel_df = pd.DataFrame(relationships)
            return rel_df
            
        except Exception as e:
            return f"Error extracting relationships: {str(e)}"
            
    def analyze_schema(self):
        """
        Provide analysis of the database schema.
        """
        if not self.engine:
            return "Database connection required for schema analysis."
            
        try:
            # Get schema structure
            inspector = inspect(self.engine)
            schema_data = []
            
            # Get all schemas
            query = text("SELECT DISTINCT TABLE_SCHEMA FROM INFORMATION_SCHEMA.TABLES ORDER BY TABLE_SCHEMA")
            schemas = [row[0] for row in self.connection.execute(query)]
            
            for schema in schemas:
                tables = inspector.get_table_names(schema=schema)
                for table in tables:
                    columns = inspector.get_columns(table, schema=schema)
                    schema_data.append({
                        'schema': schema,
                        'table': table,
                        'columns': len(columns)
                    })
            
            # Create summary of schema
            schema_df = pd.DataFrame(schema_data)
            schema_summary = schema_df.groupby('schema').agg(
                table_count=('table', 'count'),
                avg_columns=('columns', 'mean')
            ).reset_index()
            
            return schema_summary
            
        except Exception as e:
            return f"Error analyzing schema: {str(e)}"

# Example usage
if __name__ == "__main__":
    rag_system = SQLDatabaseRAG()
    rag_system.run_interactive_session()