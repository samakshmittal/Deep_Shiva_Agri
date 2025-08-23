//To check if data is loaded or not
MATCH (n) return n limit 10;



//Load CSV
LOAD CSV WITH HEADERS FROM 'file:///agriculture_key_stats.csv' AS row
WITH 
    trim(row.Indicator) AS indicatorName,
    trim(row.Year) AS yearValue,
    trim(row.Unit) AS unitName,
    CASE 
        WHEN row.Value CONTAINS '%' THEN toFloat(replace(row.Value, '%', ''))
        ELSE toFloat(row.Value)
    END AS valueAmount

// Merge core nodes
MERGE (i:Indicator {name: indicatorName})
MERGE (y:Year {value: yearValue})
MERGE (u:Unit {name: unitName})

// Merge or create value node uniquely for this indicator+year
MERGE (i)-[:RECORDED_IN]->(y)
MERGE (i)-[:MEASURED_IN]->(u)
MERGE (val:Value {amount: valueAmount, indicator: indicatorName, year: yearValue})
MERGE (i)-[:HAS_VALUE]->(val);




// All indicators in DB
MATCH (i:Indicator) RETURN i.name LIMIT 10;

// Foodgrains production for all years
MATCH (i:Indicator {name: "Foodgrains Production (Total)"})-[:HAS_VALUE]->(v)
RETURN v.year, v.amount;

// All indicators with percentage unit
MATCH (i:Indicator)-[:MEASURED_IN]->(u:Unit {name: "%"})
RETURN i.name LIMIT 10;



//Graph representation
MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50;

