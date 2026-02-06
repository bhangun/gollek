.PHONY: error-codes ci

error-codes:
	./scripts/generate-error-codes.sh

ci:
	CI=true mvn -pl core/golek-spi -am -DskipTests generate-resources
