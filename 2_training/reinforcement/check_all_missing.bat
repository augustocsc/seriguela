@echo off
REM Check missing configs across all 6 instances
REM Usage: check_all_missing.bat

echo Starting all 6 instances...
aws ec2 start-instances --instance-ids i-0ab8277c5128ef303 i-0dcb39ad7278622ec i-00d7e518d26082914 i-0aeeb70b76c5dc7d8 i-073564e75558da6f3 i-09aadd345995e5611

echo Waiting 40 seconds for instances to start...
timeout /t 40 /nobreak

echo Getting instance IPs...
FOR /F "tokens=1,2" %%A IN ('aws ec2 describe-instances --instance-ids i-0ab8277c5128ef303 i-0dcb39ad7278622ec i-00d7e518d26082914 i-0aeeb70b76c5dc7d8 i-073564e75558da6f3 i-09aadd345995e5611 --query "Reservations[*].Instances[*].[PublicIpAddress,Tags[?Key=='Name'].Value|[0]]" --output text') DO (
    echo.
    echo ============================================================
    echo Checking %%B at %%A
    echo ============================================================
    python check_missing_configs.py --ssh ubuntu@%%A --output missing_%%B.json
)

echo.
echo ============================================================
echo Stopping all instances...
echo ============================================================
aws ec2 stop-instances --instance-ids i-0ab8277c5128ef303 i-0dcb39ad7278622ec i-00d7e518d26082914 i-0aeeb70b76c5dc7d8 i-073564e75558da6f3 i-09aadd345995e5611

echo.
echo Done! Check missing_*.json files for results.
