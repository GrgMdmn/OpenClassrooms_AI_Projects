-- 1) 3 months old or less orders received more than 3 days after expected date.
WITH last_date AS (
  SELECT MAX(MAX(
      order_purchase_timestamp,
      order_approved_at,
      order_delivered_carrier_date,
      order_delivered_customer_date,
      order_estimated_delivery_date
  )) AS max_date
  FROM orders
)
SELECT *
FROM orders
WHERE order_status NOT IN ('unavailable', 'canceled')
  AND (julianday((SELECT max_date FROM last_date)) - julianday(order_purchase_timestamp)) < 90
  AND julianday(order_delivered_customer_date) - julianday(order_estimated_delivery_date) >= 3;
  
  
-- 2) Sellers generating sales of over 100000 reals
-- assuming : price does not have to be multiplied with order_item_id
select seller_id, sum(price)
from order_items
group by seller_id
HAVING sum(price) > 100000;


-- 3) Who are the new sellers (less than 3 months old) who
-- are already very committed to the platform (having already sold more than 30
-- products)?
-- assuming :
--	order_item_id counts the number of items ordered
--	using order_purchase_timestamp is the only way to find a seller first activity
WITH last_date AS (
  SELECT MAX(MAX(
      order_purchase_timestamp,
      order_approved_at,
      order_delivered_carrier_date,
      order_delivered_customer_date,
      order_estimated_delivery_date
  )) AS max_date
  FROM orders
),
first_sale_dates AS (
  SELECT seller_id, MIN(order_purchase_timestamp) as first_sale
  FROM orders o
  JOIN order_items oi ON o.order_id = oi.order_id
  GROUP BY seller_id
)
SELECT seller_id, sum(order_item_id)
FROM order_items oi
WHERE seller_id IN (
    SELECT seller_id
    FROM first_sale_dates
    WHERE (julianday((SELECT max_date FROM last_date)) - julianday(first_sale)) < 90
)
GROUP BY seller_id
HAVING sum(order_item_id) > 30;


-- 4) What are the 5 zip codes with:
-- more than 30 reviews, 
-- the worst average review score over the last 12 months?
-- assuming :
--	zip_code is about the sellers
WITH last_date AS (
  SELECT MAX(MAX(
      order_purchase_timestamp,
      order_approved_at,
      order_delivered_carrier_date,
      order_delivered_customer_date,
      order_estimated_delivery_date
  )) AS max_date
  FROM orders
)
SELECT 
    s.seller_zip_code_prefix AS zip_code,
    AVG(r.review_score) AS mean_reviews_score,
    COUNT(r.review_id) AS reviews_count
FROM sellers s
JOIN order_items oi ON s.seller_id = oi.seller_id
JOIN order_reviews r ON oi.order_id = r.order_id
WHERE julianday((SELECT max_date FROM last_date)) - julianday(r.review_creation_date) <= 365
GROUP BY s.seller_zip_code_prefix
HAVING reviews_count > 30
ORDER BY mean_reviews_score ASC
LIMIT 5;