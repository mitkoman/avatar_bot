#!/bin/bash

echo "🤖  Building and Deploying Avatar Travel Bot"
echo "============================================================="

REGISTRY="travelbotacr.azurecr.io"
IMAGE_NAME="travel-avatar"
TAG="latest"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
WEBAPP_NAME="avatartravel"
RESOURCE_GROUP="ai"
SUBSCRIPTION="1fa44ca8-b012-44b3-9618-da63389b9733"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build Docker image
echo ""
echo "📦 Building Docker image for linux/amd64..."
docker build --platform linux/amd64 -f "${SCRIPT_DIR}/Dockerfile.avatar" -t ${FULL_IMAGE} "${SCRIPT_DIR}"

if [ $? -ne 0 ]; then echo "❌ Docker build failed!"; exit 1; fi
echo "✅ Docker image built!"

# Push to ACR
echo ""
echo "📤 Pushing to Azure Container Registry..."
docker push ${FULL_IMAGE}

if [ $? -ne 0 ]; then echo "❌ Docker push failed!"; exit 1; fi
echo "✅ Image pushed!"

# Create webapp if it doesn't exist
echo ""
echo "🔍 Checking if webapp exists..."
EXISTING=$(az webapp show --name ${WEBAPP_NAME} --resource-group ${RESOURCE_GROUP} --subscription ${SUBSCRIPTION} --query name -o tsv 2>/dev/null)

if [ -z "${EXISTING}" ]; then
    echo "🆕 Creating new webapp ${WEBAPP_NAME}..."

    # Get App Service Plan from testtravel
    PLAN=$(az webapp show --name testtravel --resource-group ${RESOURCE_GROUP} --subscription ${SUBSCRIPTION} --query appServicePlanId -o tsv)

    az webapp create \
        --name ${WEBAPP_NAME} \
        --resource-group ${RESOURCE_GROUP} \
        --subscription ${SUBSCRIPTION} \
        --plan "${PLAN}" \
        --deployment-container-image-name ${FULL_IMAGE}

    if [ $? -ne 0 ]; then echo "❌ Webapp creation failed!"; exit 1; fi

    # Configure ACR credentials
    ACR_USER=$(az acr credential show --name travelbotacr --query username -o tsv)
    ACR_PASS=$(az acr credential show --name travelbotacr --query passwords[0].value -o tsv)

    az webapp config container set \
        --name ${WEBAPP_NAME} \
        --resource-group ${RESOURCE_GROUP} \
        --subscription ${SUBSCRIPTION} \
        --docker-custom-image-name ${FULL_IMAGE} \
        --docker-registry-server-url https://travelbotacr.azurecr.io \
        --docker-registry-server-user ${ACR_USER} \
        --docker-registry-server-password ${ACR_PASS}

    # Set environment variables
    az webapp config appsettings set \
        --name ${WEBAPP_NAME} \
        --resource-group ${RESOURCE_GROUP} \
        --subscription ${SUBSCRIPTION} \
        --settings \
            DID_API_KEY="${DID_API_KEY}" \
            APP_URL="https://${WEBAPP_NAME}.azurewebsites.net" \
            AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY}" \
            AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT}" \
            AZURE_OPENAI_DEPLOYMENT_NAME="${AZURE_OPENAI_DEPLOYMENT_NAME:-gpt-4o-mini}" \
            AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-08-01-preview}" \
            GOOGLE_MAPS_API_KEY="${GOOGLE_MAPS_API_KEY}" \
            TRIPADVISOR_API_KEY="${TRIPADVISOR_API_KEY}" \
            AMADEUS_CLIENT_ID="${AMADEUS_CLIENT_ID}" \
            AMADEUS_CLIENT_SECRET="${AMADEUS_CLIENT_SECRET}" \
            TAVILY_API_KEY="${TAVILY_API_KEY}" \
            WEBSITES_PORT="8000"

    # Enable Always On
    az webapp config set \
        --name ${WEBAPP_NAME} \
        --resource-group ${RESOURCE_GROUP} \
        --subscription ${SUBSCRIPTION} \
        --always-on true

    echo "✅ Webapp created and configured!"
else
    echo "✅ Webapp already exists — restarting..."
    az webapp restart --name ${WEBAPP_NAME} --resource-group ${RESOURCE_GROUP} --subscription ${SUBSCRIPTION}
    if [ $? -ne 0 ]; then echo "❌ Webapp restart failed!"; exit 1; fi
    echo "✅ Webapp restarted!"
fi

echo ""
echo "============================================================="
echo "🎉 Deployment Complete!"
echo "============================================================="
echo ""
echo "Image : ${FULL_IMAGE}"
echo "Webapp: https://${WEBAPP_NAME}.azurewebsites.net/"
echo ""
echo "============================================================="
