-- each check returns (name, number_of_violations); 0 = pass
SELECT 'fact_cpi has >= 1000 rows', CASE WHEN count(*) >= 1000 THEN 0 ELSE 1 END FROM fact_cpi
UNION ALL SELECT 'no duplicate (region,item,month)', count(*) FROM (SELECT 1 FROM fact_cpi GROUP BY region_id,item_id,date_id HAVING count(*)>1)
UNION ALL SELECT 'no null CPI', count(*) FROM fact_cpi WHERE cpi IS NULL
UNION ALL SELECT 'CPI positive', count(*) FROM fact_cpi WHERE cpi <= 0
UNION ALL SELECT 'every region has every month', count(*) FROM (SELECT region_id FROM fact_cpi WHERE item_id=(SELECT item_id FROM dim_item WHERE is_headline) GROUP BY region_id HAVING count(*) <> (SELECT count(*) FROM dim_date))
UNION ALL SELECT 'national headline weight = 100', count(*) FROM fact_weight w JOIN dim_item i USING(item_id) JOIN dim_region r USING(region_id) WHERE i.is_headline AND r.is_national AND abs(w.weight_national-100)>0.01
UNION ALL SELECT 'regional headline shares sum to 100', CASE WHEN abs((SELECT sum(weight_national) FROM fact_weight w JOIN dim_item i USING(item_id) JOIN dim_region r USING(region_id) WHERE i.is_headline AND NOT r.is_national)-100)<0.05 THEN 0 ELSE 1 END
UNION ALL SELECT 'in-region weights sum to 100 per region', count(*) FROM (SELECT region_id FROM fact_weight w JOIN dim_item i USING(item_id) WHERE NOT i.is_headline GROUP BY region_id HAVING abs(sum(weight_in_region)-100)>0.05)
UNION ALL SELECT 'yoy missing only in first 12 months', count(*) FROM fact_cpi WHERE yoy_pct IS NULL AND date_id >= 201901;
